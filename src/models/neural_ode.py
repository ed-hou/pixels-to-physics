"""
This leans dq/dt and dp/dt as a black box mlp with no prebaked physics constraints/paramaters.
It is trained on state_t, state{d+dt} paris from syntehtic trajectories. We will attempt to see energy drift over long time horizons, which will
motive the HNN.
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """
    ReLU would produce nonsmooth phase portrains and discontinuous gradients. Using Tanh activations with width
    64. Enough for the SHO/pendulum problem
    """

    def __init__(self, state_dim=2, hidden=64):
        super().__init__()
        self.net= nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, state_dim),)

    def forward(self, t, state):
        return self.net(state)

class OGNeuralODE(nn.Module):
    """
    wraps ODE func with an integrate().
    Underlying ODEfunc compared to the HNN's HamiltonianFunc which will share the same interface
    """
    def __init__(self, state_dim=2, hidden=64, solver = 'dopri5'):
        super().__init__()
        self.func = ODEFunc(state_dim=state_dim, hidden=hidden)
        self.solver = solver

    def integrate(self, state0, t_eval):
        return odeint(self.func, state0, t_eval, method=self.solver, rtol=1e-5, atol=1e-7, options={'dtype': torch.float32})


