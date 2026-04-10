import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Prioritized Experience Replay (PER) via SumTree
# ==========================================

class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    Used for efficient $O(\log N)$ sampling and updating in Prioritized Experience Replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Experience Replay Buffer prioritizing transitions with high Temporal Difference (TD) error.
    """
    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 0.01  # Small constant to prevent zero priority
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total_priority / batch_size
        priorities = []

        # Anneal beta towards 1.0
        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # Calculate Importance Sampling (IS) weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        is_weights = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # Normalize for stability

        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards).reshape(-1, 1), 
                np.array(next_states), np.array(dones).reshape(-1, 1), idxs, is_weights)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            priority = (abs(err) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

# ==========================================
# 2. DDPG Network Architectures
# ==========================================

class Actor(nn.Module):
    """
    Actor Network: Maps state to continuous deterministic actions.
    Architecture: State -> 256 -> 128 -> 64 -> Action (Ref: Table 2).
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        self.fc4 = nn.Linear(64, action_dim)
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Orthogonal initialization for improved deep RL stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Final layer uses a very small uniform distribution to avoid premature tanh saturation
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)

    def forward(self, state):
        x = F.silu(self.ln1(self.fc1(state)))  # SiLU (Swish) as used in modern networks
        x = F.silu(self.ln2(self.fc2(x)))
        x = F.silu(self.ln3(self.fc3(x)))
        action = torch.tanh(self.fc4(x))       # Bounded action space [-1, 1]
        return action

class Critic(nn.Module):
    """
    Critic Network: Evaluates the Q-value given a state-action pair.
    Architecture: State + Action -> 256 -> 128 -> 1 (Ref: Table 2).
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.silu(self.ln1(self.fc1(x)))
        x = F.silu(self.ln2(self.fc2(x)))
        q_value = self.fc3(x)
        return q_value

# ==========================================
# 3. Exploration Noise
# ==========================================

class DecayingGaussianNoise:
    """Provides Gaussian noise that decays over episodes to transition from exploration to exploitation."""
    def __init__(self, action_dim, init_sigma=0.1, decay_rate=0.99, min_sigma=0.01):
        self.action_dim = action_dim
        self.sigma = init_sigma
        self.decay_rate = decay_rate
        self.min_sigma = min_sigma

    def sample(self):
        noise = np.random.normal(0, self.sigma, size=self.action_dim)
        self.sigma = max(self.min_sigma, self.sigma * self.decay_rate)
        return noise

# ==========================================
# 4. Core DDPG Agent
# ==========================================

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) Agent tailored for Hyperparameter Tuning.
    Utilizes PyTorch optimizations, AdamW, and soft target updates.
    """
    def __init__(self, state_dim=12, action_dim=4, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize Networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers (AdamW for superior weight decay handling)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=1e-4, weight_decay=1e-5)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Memory & Noise
        self.memory = PrioritizedReplayBuffer(capacity=50000)
        self.noise = DecayingGaussianNoise(action_dim, init_sigma=0.15, decay_rate=0.995)
        
        # Hyperparameters
        self.gamma = 0.99   # Discount factor
        self.tau = 0.005    # Soft update parameter
        self.batch_size = 64

    def select_action(self, state, add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        self.actor.train()
        
        if add_noise:
            action += self.noise.sample()
            
        return np.clip(action, -1.0, 1.0)

    def train_step(self):
        """Execute one step of network weight updates using sampled batches from PER."""
        if self.memory.write_idx < self.batch_size and self.memory.tree.total_priority == 0:
            return 0.0, 0.0  # Insufficient samples

        # Sample from Prioritized Replay Buffer
        states, actions, rewards, next_states, dones, idxs, is_weights = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # ================= Update Critic =================
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            expected_q = rewards + (1.0 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        td_errors = torch.abs(expected_q - current_q).detach().cpu().numpy()
        
        # Compute Critic Loss weighted by IS weights
        critic_loss = (is_weights * F.mse_loss(current_q, expected_q, reduction='none')).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0) # Gradient clipping
        self.critic_optimizer.step()

        # Update PER tree priorities based on new TD errors
        self.memory.update_priorities(idxs, td_errors)

        # ================= Update Actor =================
        # Maximize Q-value -> Minimize -Q
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ================= Soft Update Target Networks =================
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

# ==========================================
# 5. Environment Wrapper for DL Model Tuning
# ==========================================

class ModelTrainingEnvironment:
    """
    Abstracts the deep learning model training process into an RL environment.
    Handles the conversion of DDPG actions to model hyperparameters and calculates rewards.
    """
    def __init__(self):
        self.state_dim = 12
        self.action_dim = 4
        self.action_history = deque(maxlen=5) # Buffer to calculate Var(A_t)
        self.prev_mse = None
        
    def reset(self):
        self.action_history.clear()
        self.prev_mse = 100.0  # Initial high pseudo-error
        # State: [Validation_MSE, Physics_Residual, Gradient_Norm, ..., Prev_Actions]
        return np.zeros(self.state_dim)

    def decode_action(self, action_raw):
        """
        Maps the raw neural network output [-1, 1] to the physical hyperparameter ranges (Table 2).
        """
        # 1. Physics Loss Weight (lambda): [0.1, 5]
        lambda_phys = 0.1 + (action_raw[0] + 1.0) / 2.0 * (5.0 - 0.1)
        
        # 2. Learning Rate (eta_lr): [1e-5, 1e-3]
        # Mapped in LOGARITHMIC space for smooth and uniform search
        log_lr = np.log10(1e-5) + (action_raw[1] + 1.0) / 2.0 * (np.log10(1e-3) - np.log10(1e-5))
        lr = 10 ** log_lr
        
        # 3. Dropout Rate (beta_dropout): [0.1, 0.5]
        dropout = 0.1 + (action_raw[2] + 1.0) / 2.0 * (0.5 - 0.1)
        
        # 4. Attention Temperature (gamma_attn): [0.1, 1.0]
        attn_gamma = 0.1 + (action_raw[3] + 1.0) / 2.0 * (1.0 - 0.1)
        
        return {
            'lambda_physics': lambda_phys, 
            'learning_rate': lr, 
            'dropout_rate': dropout, 
            'attn_temperature': attn_gamma
        }

    def step(self, action_raw, current_mse, smooth_metric):
        """
        Calculates the composite reward function based on Eq. (16).
        R_t = 10 * (1 - MSE_t/MSE_t-1) + 2 * Smooth - 0.5 * Var(A_t)
        """
        self.action_history.append(action_raw)
        
        # Term 1: Accuracy Improvement Reward
        if self.prev_mse is None or self.prev_mse == 0:
            acc_gain = 0.0
        else:
            acc_gain = 10.0 * (1.0 - current_mse / self.prev_mse)
            
        # Term 3: Action Fluctuation Penalty (Var(A_t))
        if len(self.action_history) > 1:
            action_var = np.var(self.action_history, axis=0).mean()
        else:
            action_var = 0.0
        
        # Composite Reward
        reward = acc_gain + 2.0 * smooth_metric - 0.5 * action_var
        
        # Update baseline for next step
        self.prev_mse = current_mse
        
        # Generate subsequent state (In a real scenario, extract metrics from PyTorch model)
        next_state = np.random.randn(self.state_dim) 
        done = False 
        
        return next_state, float(reward), done

# ==========================================
# 6. Main Orchestration Loop
# ==========================================

def main_training_loop():
    """
    Demonstrates the interaction between the RL Agent (DDPG) and the Environment (DL Model).
    """
    env = ModelTrainingEnvironment()
    agent = DDPGAgent(state_dim=12, action_dim=4, device='cuda')
    
    total_epochs = 200
    update_freq = 5  # DDPG hyperparameter update frequency (as per paper)
    
    state = env.reset()
    print("====== Initiating RL-Driven Adaptive Training ======\n")
    
    for epoch in range(1, total_epochs + 1):
        # 1. RL Agent selects an action (Hyperparameter set)
        action_raw = agent.select_action(state, add_noise=True)
        hyperparams = env.decode_action(action_raw)
        
        if epoch % update_freq == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Applied Hyperparams: "
                  f"LR={hyperparams['learning_rate']:.2e}, "
                  f"Phys_Weight={hyperparams['lambda_physics']:.2f}, "
                  f"Dropout={hyperparams['dropout_rate']:.2f}, "
                  f"Attn_Temp={hyperparams['attn_temperature']:.2f}")
        
        # ---------------------------------------------------------
        # [PLACEHOLDER FOR DEEP LEARNING MODEL TRAINING LOGIC]
        # 1. Update optimizer: optimizer.param_groups[0]['lr'] = hyperparams['learning_rate']
        # 2. Update network bounds: model.dropout.p = hyperparams['dropout_rate']
        # 3. Train DL model for 1 epoch
        # 4. Calculate DL Validation MSE & Sequence Smoothness
        # ---------------------------------------------------------
        
        # Simulate realistic diminishing MSE for demonstration
        simulated_current_mse = np.exp(-epoch / 40.0) + np.random.normal(0, 0.02) 
        simulated_smoothness = 1.0 # Assume ideal smoothness
        
        # DDPG Environment Step and Network Update (Every 5 epochs)
        if epoch % update_freq == 0:
            next_state, reward, done = env.step(action_raw, simulated_current_mse, simulated_smoothness)
            
            # Store transition in Prioritized Replay Buffer
            agent.memory.push(state, action_raw, reward, next_state, done)
            
            # Update DDPG networks (Multiple passes for higher sample efficiency)
            c_loss, a_loss = 0, 0
            for _ in range(3): 
                c_l, a_l = agent.train_step()
                c_loss += c_l
                a_loss += a_l
                
            print(f"          -> Agent Reward: {reward:+.3f} | Critic Loss: {c_loss/3:.3f} | Actor Loss: {a_loss/3:.3f}")
            state = next_state
            
        if done:
            break
            
    print("\n====== Training Completed Successfully ======")

if __name__ == "__main__":
    main_training_loop()
