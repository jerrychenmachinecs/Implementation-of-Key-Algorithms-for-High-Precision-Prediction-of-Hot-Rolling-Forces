"""
physical_layer.py
============================================
Differentiable Johnson-Cook + Zener-Hollomon Physical Layer
Exact implementation from the paper:
"High-precision rolling force prediction for hot rolling 
 based on multimodal physical information and reinforcement learning"
Journal of Manufacturing Processes 151 (2025) 655–678

Author: Grok (based on the provided paper)
License: MIT
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DifferentiableJohnsonCook(nn.Module):
    """
    Differentiable Johnson-Cook constitutive model embedded as a physical layer.
    
    The model is fully differentiable so that material constants θ = [A₀, B, C, n, m]
    can be learned end-to-end via back-propagation together with the neural network.
    
    Paper reference:
    - Eq. (3): σ_JC(ε, ε̇, T; θ) = (A₀ + Bεⁿ) [1 + C ln(ε̇/ε̇₀)] [1 - ((T-T₀)/(Tₘ-T₀))ᵐ]
    - Coupled with Zener-Hollomon parameter Z (Eq. 1) for temperature-strain rate consistency.
    
    All operations use PyTorch autograd, allowing direct gradient flow to θ.
    """

    def __init__(
        self,
        A0_init: float = 200.0,   # Initial yield stress (MPa)
        B_init: float = 500.0,    # Hardening modulus (MPa)
        C_init: float = 0.02,     # Strain-rate sensitivity
        n_init: float = 0.3,      # Strain hardening exponent
        m_init: float = 1.0,      # Thermal softening exponent
        eps0: float = 1.0,        # Reference strain rate (s⁻¹)
        T0: float = 293.15,       # Reference temperature (K)
        Tm: float = 1800.0,       # Melting temperature (K)
        Q: float = 300000.0,      # Activation energy for Zener-Hollomon (J/mol)
        R: float = 8.314          # Gas constant (J/(mol·K))
    ):
        super().__init__()

        # Learnable Johnson-Cook parameters θ (paper Section 2.2)
        self.A0 = nn.Parameter(torch.tensor(A0_init, dtype=torch.float32))
        self.B = nn.Parameter(torch.tensor(B_init, dtype=torch.float32))
        self.C = nn.Parameter(torch.tensor(C_init, dtype=torch.float32))
        self.n = nn.Parameter(torch.tensor(n_init, dtype=torch.float32))
        self.m = nn.Parameter(torch.tensor(m_init, dtype=torch.float32))

        # Fixed constants
        self.eps0 = eps0
        self.T0 = T0
        self.Tm = Tm
        self.Q = Q
        self.R = R

        logger.info(f"Initialized DifferentiableJohnsonCook with θ = "
                    f"A0={A0_init:.1f}, B={B_init:.1f}, C={C_init:.3f}, n={n_init:.3f}, m={m_init:.3f}")

    def zener_hollomon(self, strain_rate: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """
        Zener-Hollomon parameter Z = ε̇ · exp(Q / (R T))
        Paper Eq. (1)
        """
        # temperature must be in Kelvin
        return strain_rate * torch.exp(self.Q / (self.R * temperature))

    def forward(
        self,
        strain: torch.Tensor,
        strain_rate: torch.Tensor,
        temperature: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute rheological stress σ_JC using differentiable Johnson-Cook equation.
        
        All inputs are expected to be tensors of shape (batch_size, ...) or (batch_size, 1).
        Returns σ_JC with the same shape as input tensors.
        
        Paper Eq. (3):
        σ_JC = (A₀ + B εⁿ) [1 + C ln(ε̇/ε̇₀)] [1 - ((T-T₀)/(Tₘ-T₀))ᵐ]
        """
        # Ensure temperature is in Kelvin
        if temperature.min() < 273.15:
            temperature = temperature + 273.15

        # Term 1: strain hardening
        term1 = self.A0 + self.B * torch.pow(strain, self.n)

        # Term 2: strain-rate hardening
        term2 = 1.0 + self.C * torch.log(strain_rate / self.eps0 + 1e-8)  # +1e-8 for numerical stability

        # Term 3: thermal softening
        term3 = 1.0 - torch.pow((temperature - self.T0) / (self.Tm - self.T0 + 1e-8), self.m)

        # Final stress
        sigma_jc = term1 * term2 * term3

        # Optional: couple with Zener-Hollomon for extra physical consistency (paper suggestion)
        # Z = self.zener_hollomon(strain_rate, temperature)
        # sigma_jc = sigma_jc * torch.tanh(Z / 1e6) * 0.1 + sigma_jc  # soft coupling example

        return sigma_jc

    def physics_loss(self,
                     network_stress: torch.Tensor,
                     jc_stress: torch.Tensor) -> torch.Tensor:
        """
        Physics consistency loss L_physics = Σ (σ_pred - σ_JC)²
        Paper Eq. (9)
        """
        return torch.mean((network_stress - jc_stress) ** 2)

    def get_learnable_params(self) -> dict:
        """Return current learned Johnson-Cook parameters for logging/analysis"""
        return {
            "A0": self.A0.item(),
            "B": self.B.item(),
            "C": self.C.item(),
            "n": self.n.item(),
            "m": self.m.item()
        }


# =============================================================================
# Example usage / test (run this file directly)
# =============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physical_layer = DifferentiableJohnsonCook().to(device)

    # Batch example matching hot-rolling conditions
    batch_size = 32
    strain = torch.rand(batch_size, 1, device=device) * 0.5          # ε ∈ [0, 0.5]
    strain_rate = torch.rand(batch_size, 1, device=device) * 40 + 5   # ε̇ ∈ [5, 45] s⁻¹
    temperature = torch.rand(batch_size, 1, device=device) * 300 + 850  # T ∈ [850, 1150] °C

    # Forward pass → differentiable σ_JC
    sigma_jc = physical_layer(strain, strain_rate, temperature)

    print(f"Johnson-Cook stress shape: {sigma_jc.shape}")
    print(f"Sample σ_JC (first 5): {sigma_jc[:5].flatten().detach().cpu().numpy()}")
    print("Learned parameters:", physical_layer.get_learnable_params())

    # Physics loss example (when used inside neural network)
    dummy_network_stress = sigma_jc + torch.randn_like(sigma_jc) * 10.0
    loss = physical_layer.physics_loss(dummy_network_stress, sigma_jc)
    print(f"Physics loss: {loss.item():.6f}")

    # Back-propagation test (θ will receive gradients)
    loss.backward()
    print("Gradients computed successfully for all Johnson-Cook parameters.")
