"""
Hamiltonian neural net. We learn the scalar H_theta(q,p) and have hamilton's eqn derive  dq/dt = +dH/dp, dp/dt = -dH/dq.
Energy conservation is maintained by modulo integration error baked in instead of training data.
We use the same forward(t,state) interface as odefunc.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint

class HamiltonianFunc(nn.Module):
    """
    2,200,200,1 tanh activations.
    Following greydanus et al.NeurIPS 2019, we use hidden width with 200, we are wider than ODEFunc's 64 because we're learning a scalar to derive a 2dvecgtor field.
    ensuring capacity on representing H accurately and keep learned dynamics stable over long rollout horizons
    """

    def __init__(self, state_dim=2, hidden=200):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden,1))
        #normalization buffers
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))


    def set_normalization(self,mean, std):
        self.state_mean.copy_(torch.as_tensor(mean, dtype=self.state_mean.dtype))
        self.state_std.copy_(torch.as_tensor(std, dtype=self.state_std.dtype))

    def forward(self, t, state):
        # requires_grad on the unnormalized coord. autograd carries the 1/sigma chain rule through normalization
        state= state.requires_grad_(True)
        state_norm = (state-self.state_mean)/ self.state_std
        H=self.net(state_norm).sum()
        dH=torch.autograd.grad(H, state, create_graph=True)[0]
        dq_dt = dH[...,1]
        dp_dt = -dH[..., 0]
        return torch.stack([dq_dt, dp_dt], dim=-1)

class HNN(nn.Module):
     #hamiltonianfunc helper.
    def __init__(self, state_dim=2, hidden=200, solver='dopri5'):
        super().__init__()
        self.func=HamiltonianFunc(state_dim=state_dim, hidden=hidden)
        self.solver=solver

    def integrate(self, state0, t_eval):
        #need enable_grad because the forward pass derives ds/dt via autgrad through H.
        with torch.enable_grad():
            return odeint(self.func, state0, t_eval, method=self.solver, rtol=1e-5, atol=1e-7)
