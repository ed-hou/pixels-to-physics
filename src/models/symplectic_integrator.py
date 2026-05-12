"""
This is the leapfrog symplectic integrator for hnn trajectory at inference

HNN derives ds/dt via autograd thru H as dpori5 doesn't preserve the symplectic 2 form so there's also
a small energy drift. Because leapfrog is symplectic by construction, it conserves a H~ which means H_0 oscillates about H_0 with bounded amplitude forever.
I swapped H_net and replaced the integrator.
"""

import torch

def leap_step(ham_func, q, p, dt):
    """
    symplectic steps are as follows:
    half kick where p_{p1/2}= p-(dt/2)*dH/dq eval at (q,p)
    full drift where q_new = q+dt*dH/dp eval at (q,p_{1/2}
    half kick where p_new = p_{1/2} - (dt/2)*dH/dq eval at (q_new, p+{1/2}
    w/ each kick recomputing dH/dq @current state
    """
    n=q.shape[-1]
    q=q.detach()
    p=p.detach()

    #halfkick1
    current_state = torch.cat([q,p], dim=-1). requires_grad_(True)
    H= ham_func.net((current_state- ham_func.state_mean)/ ham_func.state_std).sum()
    dH=torch.autograd.grad(H,current_state,create_graph=False)[0]
    dHdp = dH[..., :n]
    p_half = (p-.5*dt*dHdp).detach()

    #full drift
    current_state = torch.cat([q,p_half], dim=-1). requires_grad_(True)
    H=ham_func.net((current_state- ham_func.state_mean) / ham_func.state_std).sum()
    dH=torch.autograd.grad(H, current_state, create_graph=False)[0]
    dHdq_half = dH[..., n:]
    q_new = (q + dt*dHdq_half).detach()

    #halfkick2
    current_state=torch.cat([q_new,p_half], dim=-1). requires_grad_(True)
    H= ham_func.net((current_state- ham_func.state_mean) / ham_func.state_std).sum()
    dH=torch.autograd.grad(H, current_state, create_graph=False)[0]
    dHdq_new = dH[..., :n]
    p_new = (p_half-.5*dt*dHdq_new).detach()

    return q_new, p_new

def leapfrog_integrate(ham_func, state0, t_eval):

    #integrate the HNN symplectically over T_eval.
    n=state0.shape[-1]//2
    q=state0[..., :n].clone()
    p=state0[..., n:].clone()

    traj=[torch.cat([q,p], dim =-1).detach()]
    for i in range(1, len(t_eval)):
        dt=(t_eval[i]-t_eval[i-1]).item()
        q,p=leap_step(ham_func, q, p, dt)
        traj.append(torch.cat([q,p], dim=-1).detach())

    return torch.stack(traj, dim=0)


