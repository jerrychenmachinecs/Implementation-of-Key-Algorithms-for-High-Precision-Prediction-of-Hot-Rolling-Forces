import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
from collections import deque
import warnings

# Suppress LightGBM/SciPy warnings for cleaner console output
warnings.filterwarnings('ignore')

# ==========================================
# 1. Physical Mechanism & Differentiable Layer
# ==========================================

class ZenerHollomonMechanics:
    """
    Zener-Hollomon Physical Mechanism Baseline (Ref: Eq. 19).
    Calculates the theoretical stress estimate based on material thermodynamics.
    """
    def __init__(self, Q=270000.0, R=8.314, A=1e10, n=5.0):
        self.Q = Q
        self.R = R
        self.A = A
        self.n = n

    def compute_theoretical_stress(self, strain_rate, temperature):
        # Calculate Zener-Hollomon parameter Z: Z = strain_rate * exp(Q / RT)
        # Temperature is converted from Celsius to Kelvin
        T_kelvin = temperature + 273.15
        Z = strain_rate * np.exp(self.Q / (self.R * T_kelvin))
        
        # Calculate theoretical flow stress \sigma_Z
        sigma_z = (Z / self.A) ** (1.0 / self.n)
        return sigma_z

class DifferentiableJohnsonCook(nn.Module):
    """
    Differentiable Johnson-Cook Constitutive Equation.
    Designed for Type-II anomalies to enable online incremental learning via PyTorch Autograd.
    """
    def __init__(self, A0=250.0, B=100.0, C=0.015, n=0.5, m=1.0, T_m=1500.0, T_0=20.0):
        super().__init__()
        # Define parameters as learnable nn.Parameter for gradient-based updates
        self.A0 = nn.Parameter(torch.tensor(A0, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(B, dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor(C, dtype=torch.float32))
        self.n = nn.Parameter(torch.tensor(n, dtype=torch.float32))
        self.m = nn.Parameter(torch.tensor(m, dtype=torch.float32))
        
        self.T_m = T_m
        self.T_0 = T_0
        self.eps_dot_0 = 1.0

    def forward(self, eps, eps_dot, T):
        # Strict differentiable implementation of the Johnson-Cook equation (Eq. 12)
        term1 = self.A0 + self.B * (eps ** self.n)
        term2 = 1.0 + self.C * torch.log(eps_dot / self.eps_dot_0)
        term3 = 1.0 - ((T - self.T_0) / (self.T_m - self.T_0)) ** self.m
        sigma_jc = term1 * term2 * term3
        return sigma_jc

# ==========================================
# 2. Dual-Drive Anomaly Detector
# ==========================================

class DualDriveDetector:
    """
    Residual-Physical Dual-Driven Anomaly Detection System.
    Combines MC-Dropout data-driven uncertainty with Zener-Hollomon physical consistency.
    """
    def __init__(self, window_size=50):
        self.zh_model = ZenerHollomonMechanics()
        self.residual_buffer = deque(maxlen=window_size)
        
        # Dynamic adaptive parameters (Ref: Eq. 18 & 23)
        self.k_t = 2.58       # Initial confidence interval coefficient (~99%)
        self.sigma_eps = 5.0  # Measurement noise standard deviation
        self.alpha_k = 0.05   # Control factor for k_t self-regulation

    def mc_dropout_inference(self, x_process):
        """
        Simulate Monte Carlo Dropout inference to obtain mean prediction and epistemic uncertainty.
        In production, this calls the actual Multi-channel CNN-RNN model with N stochastic forward passes.
        """
        # Mock prediction for simulation purposes
        base_pred = 15000.0 + x_process['speed'] * 100.0 
        mc_preds = np.random.normal(loc=base_pred, scale=200.0, size=30)
        return np.mean(mc_preds), np.var(mc_preds)

    def detect(self, x_process, y_true):
        # 1. Data-Driven Residual Monitoring (Eq. 17 & 18)
        p_hat, var_t = self.mc_dropout_inference(x_process)
        residual = y_true - p_hat
        self.residual_buffer.append(residual)

        mu_hist = np.mean(self.residual_buffer) if len(self.residual_buffer) > 10 else 0.0
        dynamic_threshold = abs(mu_hist) + self.k_t * np.sqrt(var_t) + self.sigma_eps**2
        
        data_anomaly = abs(residual) > dynamic_threshold

        # 2. Physical Consistency Check (Eq. 19)
        eps_dot = x_process['strain_rate']
        temp = x_process['temperature']
        
        # Approximate conversion from force (kN) to stress (MPa) for physical evaluation
        sigma_pred = p_hat / 100.0 
        sigma_z = self.zh_model.compute_theoretical_stress(eps_dot, temp)
        
        # Calculate physical residual ratio R_physics
        r_physics = abs(sigma_pred - sigma_z) / (sigma_z + 1e-6)
        physics_anomaly = r_physics > 0.3  # Threshold specified in the paper

        # 3. Dual-Criteria Final Judgment
        is_anomaly = data_anomaly and physics_anomaly
        
        return is_anomaly, p_hat, residual, r_physics

    def update_k_t(self, n_valid, n_alarm):
        """Self-regulation of confidence interval coefficient (Eq. 23)."""
        n_total = n_valid + n_alarm + 1e-5
        # Exponential adjustment based on false alarm / valid operation ratio
        self.k_t = self.k_t * np.exp(-self.alpha_k * ((n_valid - n_alarm) / n_total))
        self.k_t = np.clip(self.k_t, 1.5, 4.0)  # Constrain bounds to prevent collapse

# ==========================================
# 3. LightGBM Multi-Modal Anomaly Classifier
# ==========================================

class AnomalyClassifier:
    """
    Gradient Boosting Tree (LightGBM) model to classify the specific type of anomaly 
    using multi-source features (Table 3).
    """
    def __init__(self):
        self.model = lgb.Booster()
        self.is_trained = False

    def extract_features(self, state_history):
        """Extract multi-source features as defined in Table 3."""
        residuals = [s['residual'] for s in state_history]
        speeds = [s['speed'] for s in state_history]
        temps = [s['temperature'] for s in state_history]
        
        features = {
            'res_skew': skew(residuals) if len(residuals) > 2 else 0.0,
            'res_kurtosis': kurtosis(residuals) if len(residuals) > 2 else 0.0,
            'speed_jump_rate': np.max(np.abs(np.diff(speeds))) if len(speeds) > 1 else 0.0,
            'temp_drop': np.max(temps) - np.min(temps) if len(temps) > 0 else 0.0,
            'physics_dev': state_history[-1]['r_physics'],
            'stiffness_dev': state_history[-1]['stiffness'] - state_history[-1]['nominal_stiffness']
        }
        return np.array(list(features.values())).reshape(1, -1)

    def train_dummy_model(self):
        """Train a mock LightGBM model for simulation out-of-the-box."""
        X_train = np.random.randn(1000, 6)
        y_train = np.random.randint(0, 4, 1000) # Classes: 0 (Type-I), 1 (Type-II), 2 (Type-III), 3 (Type-IV)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'multiclass', 
            'num_class': 4, 
            'verbose': -1,
            'learning_rate': 0.05
        }
        self.model = lgb.train(params, train_data, num_boost_round=50)
        self.is_trained = True

    def classify(self, state_history):
        if not self.is_trained:
            self.train_dummy_model()
            
        features = self.extract_features(state_history)
        probs = self.model.predict(features)
        anomaly_type_idx = np.argmax(probs[0])
        
        types = ['Type-I', 'Type-II', 'Type-III', 'Type-IV']
        return types[anomaly_type_idx]

# ==========================================
# 4. Incremental Feedback Correction Engine
# ==========================================

class IncrementalCorrectionEngine:
    """
    Executes specific correction strategies based on the anomaly type identified.
    Includes Kalman filtering, online PyTorch backpropagation, and Model Predictive Control (MPC).
    """
    def __init__(self):
        # Type-I: Kalman Filter States (Eq. 20)
        self.P_hat = 15000.0
        self.P_minus = 15000.0
        self.P_err_cov = 1000.0
        self.Q_noise = 10.0   # Process noise covariance
        self.R_noise = 500.0  # Measurement noise covariance
        
        # Type-II: Differentiable Physics Model
        self.jc_model = DifferentiableJohnsonCook()
        self.jc_optimizer = torch.optim.Adam(self.jc_model.parameters(), lr=0.05)

    def correct_type_1(self, z_measured):
        """Type-I Sensor Drift: Kalman Filter Observation Reconstruction (Eq. 20)."""
        # Predict Step
        self.P_minus = self.P_hat
        P_err_minus = self.P_err_cov + self.Q_noise
        
        # Update Step
        K = P_err_minus / (P_err_minus + self.R_noise) # Kalman Gain
        self.P_hat = self.P_minus + K * (z_measured - self.P_minus)
        self.P_err_cov = (1 - K) * P_err_minus
        
        return self.P_hat

    def correct_type_2(self, actual_force, eps, eps_dot, temp):
        """Type-II Material Performance Deviation: Online Parameter Adjustment (Eq. 21)."""
        self.jc_model.train()
        
        # Convert inputs to PyTorch tensors
        eps_t = torch.tensor(eps, dtype=torch.float32)
        eps_dot_t = torch.tensor(eps_dot, dtype=torch.float32)
        temp_t = torch.tensor(temp, dtype=torch.float32)
        
        # Target stress (converted from measured force)
        target_sigma = torch.tensor(actual_force / 100.0, dtype=torch.float32) 
        
        # Forward and backward pass
        self.jc_optimizer.zero_grad()
        pred_sigma = self.jc_model(eps_t, eps_dot_t, temp_t)
        
        # Loss = MSE + L2 Regularization to prevent drastic parameter shifts
        mse_loss = nn.MSELoss()(pred_sigma, target_sigma)
        l2_lambda = 0.001
        l2_reg = sum(p.pow(2.0).sum() for p in self.jc_model.parameters())
        total_loss = mse_loss + l2_lambda * l2_reg
        
        total_loss.backward()
        self.jc_optimizer.step()
        
        return {
            'A0': self.jc_model.A0.item(),
            'B': self.jc_model.B.item()
        }

    def correct_type_3_4(self, P_ref, current_u, process_model_func):
        """
        Type-III/IV Process/Equipment Anomaly: Model Predictive Control (MPC) (Eq. 22).
        Uses Sequential Least Squares Programming (SLSQP) to find optimal process adjustment.
        """
        rho = 0.5  # Weighting coefficient balancing process fluctuation and performance
        u_min, u_max = -1.5, 1.5  # Safety bounds for speed adjustment (m/s)
        
        def objective_function(delta_u):
            # min (P_{t+k} - P_ref)^2 + rho * ||\Delta u||^2
            predicted_P = process_model_func(current_u + delta_u[0])
            return (predicted_P - P_ref)**2 + rho * (delta_u[0]**2)
            
        # Execute optimization
        res = minimize(
            objective_function, 
            x0=[0.0], 
            bounds=[(u_min, u_max)],
            method='SLSQP'
        )
        
        optimal_delta_u = res.x[0]
        return optimal_delta_u

# ==========================================
# 5. Closed-Loop System Orchestrator
# ==========================================

class RollingForceClosedLoopSystem:
    """
    Main Orchestrator integrating Detection, Classification, and Correction.
    Coordinates the real-time data flow for the hot continuous rolling process.
    """
    def __init__(self):
        self.detector = DualDriveDetector()
        self.classifier = AnomalyClassifier()
        self.correction_engine = IncrementalCorrectionEngine()
        
        self.state_history = deque(maxlen=20) # Maintains historical sliding window
        
        # Global metrics for adaptive adjustments
        self.n_valid = 0
        self.n_alarm = 0

    def process_pass(self, current_pass_data):
        """Process a single rolling stand (pass)."""
        y_true = current_pass_data['actual_force']
        
        # 1. Execute Dual-Drive Anomaly Detection
        is_anomaly, p_hat, residual, r_phys = self.detector.detect(current_pass_data, y_true)
        
        # Update state history with current metrics
        current_pass_data['residual'] = residual
        current_pass_data['r_physics'] = r_phys
        self.state_history.append(current_pass_data)

        if not is_anomaly:
            self.n_valid += 1
            self.detector.update_k_t(self.n_valid, self.n_alarm)
            return {"status": "Normal", "prediction": p_hat}
            
        # 2. Handle Triggered Anomaly
        self.n_alarm += 1
        self.detector.update_k_t(self.n_valid, self.n_alarm)
        
        # 3. Classify Anomaly Type via LightGBM
        anomaly_type = self.classifier.classify(self.state_history)
        print(f"\n[ALARM] Anomaly Detected: {anomaly_type}. Initiating dynamic incremental correction...")

        # 4. Route to specific Correction Strategy
        response = {"status": "Corrected", "anomaly_type": anomaly_type}
        
        if anomaly_type == 'Type-I':
            # Sensor Drift -> Kalman Filter
            corrected_force = self.correction_engine.correct_type_1(y_true)
            response['kalman_output'] = corrected_force
            print(f"  -> Kalman Filter correction on sensor reading: {y_true:.1f} kN -> {corrected_force:.1f} kN")
            
        elif anomaly_type == 'Type-II':
            # Material Offset -> Online PyTorch AutoGrad update
            new_params = self.correction_engine.correct_type_2(
                actual_force=y_true, 
                eps=0.2, # Approximated strain
                eps_dot=current_pass_data['strain_rate'], 
                temp=current_pass_data['temperature']
            )
            response['new_jc_params'] = new_params
            print(f"  -> Online Johnson-Cook parameter update: A0={new_params['A0']:.2f}, B={new_params['B']:.2f}")
            
        elif anomaly_type in ['Type-III', 'Type-IV']:
            # Process/Equipment Overrun -> Model Predictive Control
            
            # Create a mock differentiable process model proxy for MPC
            def mock_process_model(new_speed):
                return p_hat + (new_speed - current_pass_data['speed']) * 150.0 
                
            p_target = 15000.0 # Standard target rolling force
            delta_speed = self.correction_engine.correct_type_3_4(
                P_ref=p_target, 
                current_u=current_pass_data['speed'], 
                process_model_func=mock_process_model
            )
            response['mpc_delta_speed'] = delta_speed
            print(f"  -> MPC optimization active. Process adjustment recommendation: \Delta V = {delta_speed:.3f} m/s")

        return response

# ==========================================
# 6. Industrial Simulation Loop
# ==========================================

if __name__ == "__main__":
    # Initialize the entire closed-loop system
    system = RollingForceClosedLoopSystem()
    
    print("====== 2250mm Hot Strip Mill Intelligent Control System Initialized ======\n")
    
    # Simulate a sequence of 10 rolling passes (stands)
    for pass_idx in range(1, 11):
        # Baseline normal process data
        pass_data = {
            'speed': 5.2,
            'temperature': 950.0,
            'strain_rate': 10.0,
            'stiffness': 10000.0,
            'nominal_stiffness': 10000.0,
            'actual_force': 15050.0 + np.random.normal(0, 50.0) # Normal Gaussian noise
        }
        
        # Inject Type-I Anomaly (Sensor Drift / Distortion) at Pass 4
        if pass_idx == 4:
            pass_data['actual_force'] += 1500.0 # Significant drift
            
        # Inject Type-II Anomaly (Material Performance Deviation) at Pass 8
        if pass_idx == 8:
            pass_data['temperature'] -= 80.0    # Sudden temperature drop
            pass_data['actual_force'] += 2000.0 # Material hardening response

        print(f"Processing Stand/Pass {pass_idx}...")
        
        # Execute closed-loop analysis
        result = system.process_pass(pass_data)
        
        if result['status'] == 'Normal':
            print(f"  -> Operation stable. Model Prediction: {result['prediction']:.1f} kN")
            
    print("\n====== Production Batch Completed. Transitioning to Outer Loop Updates ======")
