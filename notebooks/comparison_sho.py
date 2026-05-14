import os
import sys
import csv
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.systems.sho import SimpleHarmonicOscillator
from src.models.hamiltonian_nn import HNN
from src.models.neural_ode import OGNeuralODE
from src.models.symplectic_integrator import leapfrog_integrate
from src.models.parametric_calibration import ParametricHamiltonianFit
from src.evaluation.metrics import (energy_drift, relativel2_error)

checkpoint_hnn = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints', 'hamiltonianSHO.pt')
checkpoint_ognode = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints', 'neural_ode_sho.pt')
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', 'sho_noise000.npz')

TLONG=50.0
NPOINTS_LONG=5000
if torch.backends.mps.is_available():
    device = torch.device('mps')

else:
    device = torch.device('cpu')
print(f"using device: {device}")
torch.set_default_dtype(torch.float32)

def load_hnn():
    model=HNN().to(device).to(torch.float32)
    model.load_state_dict(torch.load(checkpoint_hnn, map_location=device))
    model.eval()
    return model

def load_ognode():
    model=OGNeuralODE().to(device).to(torch.float32)
    model.load_state_dict(torch.load(checkpoint_ognode, map_location=device))
    model.eval()
    return model

def oracle_of_truth_trajectory(q0,p0, t_eval):
    sho=SimpleHarmonicOscillator(m=1.0, k=1.0)
    t,q,p = sho.generate_trajectory(q0=q0, p0=p0, t_span=(t_eval[0], t_eval[-1]), n_points=len(t_eval))
    H=sho.hamiltonian(q,p)
    return q,p,H

def parametric_rollout(q0,p0,t_eval):
    data=np.load(data_path)
    train_idx=data['train_idx']
    q_train=data['q_clean'][train_idx].flatten()
    p_train=data['p_clean'][train_idx].flatten()
    H_train=.5*q_train**2+.5*p_train**2

    pfit=ParametricHamiltonianFit()
    result=pfit.fit_sho(q_train,p_train,H_train)
    a,b=result['a'],result['b']
    print(f"parametric fit: a={a:.6f}, b= {b:.6f}")

    def deriv(t,state):
        q,p=state
        return [2*a*p, -2*p*q]

    soln=solve_ivp(deriv, (t_eval[0], t_eval[-1]), [q0, p0], t_eval=t_eval, method= 'DOP853', rtol=1e-12, atol=1e-12)
    return soln.y[0], soln.y[1]

def kan_rollout(q0, p0, t_eval):
    a=.4998 #coef on p^2
    b=.5000 #coef on q^2

    def deriv(t, state):
        q,p=state
        return [2*a*p, -2*p*q]

    soln=solve_ivp(deriv, (t_eval[0], t_eval[-1]), [q0,p0], t_eval=t_eval, method='DOP853', rtol=1e-12, atol=1e-12)
    return soln.y[0], soln.y[1]

def ognode_rollout(model, q0, p0, t_eval):
    state0=torch.tensor([[q0,p0]], dtype=torch.float32).to(device)
    t_eval_t=torch.tensor(t_eval, dtype=torch.float32).to(device)
    traj=model.integrate(state0, t_eval_t)
    traj=traj.squeeze(1).detach().cpu().numpy()
    return traj[:,0], traj[:,1]

def hnn_dopri5rollout(model, q0, p0, t_eval):
    state0=torch.tensor([[q0,p0]], dtype=torch.float32).to(device)
    t_eval_t=torch.tensor(t_eval, dtype=torch.float32).to(device)
    with torch.enable_grad():
        traj=model.integrate(state0, t_eval_t)
    traj=traj.squeeze(1).detach().cpu().numpy()
    return traj[:,0], traj[:,1]

def hnn_leapfrog_rollout(model, q0, p0, t_eval):
    state0=torch.tensor([q0, p0], dtype=torch.float32).to(device)
    t_eval_t=torch.tensor(t_eval, dtype=torch.float32).to(device)
    traj=leapfrog_integrate(model.func, state0, t_eval_t)
    traj=traj.detach().cpu().numpy()
    return traj[:,0], traj[:,1]

def main():
    q0, p0= 1.0, .5
    sho=SimpleHarmonicOscillator(m=1.0, k=1.0)
    H0=sho.hamiltonian(np.array([q0]), np.array([p0]))[0]
    print(f"initial state: q0={q0}, p0={p0}, H0={H0:.6f}")
    t_eval=np.linspace(0, TLONG, NPOINTS_LONG)

    q_truth, p_truth, H_truth= oracle_of_truth_trajectory(q0, p0, t_eval)

    print("Parametric rollout")
    q_par, p_par= parametric_rollout(q0, p0, t_eval)
    H_par= sho.hamiltonian(q_par, p_par)

    print ("rolling out kan")
    q_kan, p_kan= kan_rollout(q0, p0, t_eval)
    H_kan= sho.hamiltonian(q_kan, p_kan)

    print("ogneuralode rollout")
    ognode=load_ognode()
    q_og, p_og = ognode_rollout(ognode, q0, p0, t_eval)
    H_og=sho.hamiltonian(q_og, p_og)

    print("DOPRI5 rollout")
    hnn=load_hnn()
    q_hnn, p_hnn=hnn_dopri5rollout(hnn, q0, p0, t_eval)
    H_hnn=sho.hamiltonian(q_hnn, p_hnn)

    print("Leapfrog rollout")
    q_lf, p_lf= hnn_leapfrog_rollout(hnn, q0, p0, t_eval)
    H_lf=sho.hamiltonian(q_lf, p_lf)

    drift_par=energy_drift(H_par)
    drift_kan=energy_drift(H_kan)
    drift_og= energy_drift(H_og)
    drift_hnn=energy_drift(H_hnn)
    drift_lf=energy_drift(H_lf)

    truth_traj=np.column_stack([q_truth, p_truth])
    l2_par=relativel2_error(np.column_stack([q_par,p_par]), truth_traj)
    l2_kan=relativel2_error(np.column_stack([q_kan, p_kan]), truth_traj)
    l2_og=relativel2_error(np.column_stack([q_og, p_og]), truth_traj)
    l2_hnn=relativel2_error(np.column_stack([q_hnn, p_hnn]), truth_traj)
    l2_lf=relativel2_error(np.column_stack([q_lf, p_lf]), truth_traj)

    print("\n....Comparison SHO Results...")
    for name, drift, l2 in [('parametric', drift_par, l2_par), ('kan', drift_kan, l2_kan), ('ogNeuralODE', drift_og, l2_og), ('HNNdopri5', drift_hnn, l2_hnn), ('HNNleapfrog', drift_lf, l2_lf),]:
        print(f"  {name:15s} drift={drift:.6%} l2={l2:.4e}")


    csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables', 'comparison_sho.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w=csv.writer(f)
        w.writerow(['model','energy_drift','traj_l2_error_50s'])
        w.writerow(['parametric',  f"{drift_par:.6%}", f"{l2_par:.6e}"])
        w.writerow(['kan',         f"{drift_kan:.6%}", f"{l2_kan:.6e}"])
        w.writerow(['ogNeuralODE', f"{drift_og:.6%}",  f"{l2_og:.6e}"])
        w.writerow(['HNNdopri5',  f"{drift_hnn:.6%}", f"{l2_hnn:.6e}"])
        w.writerow(['HNNleapfrog',f"{drift_lf:.6%}",  f"{l2_lf:.6e}"])
    print(f"saved {csv_path}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(H0, color='black', ls='--', lw=1, label=f'true H = {H0:.3f}')
    ax.plot(t_eval, H_par, color='purple', lw=1.2, label=f'parametric, drift={drift_par:.2e}')
    ax.plot(t_eval, H_kan, color='orange', lw=1.2, label=f'KAN, drift={drift_kan:.2e}')
    ax.plot(t_eval, H_og, color='gray', lw=1.0, alpha=.7, label=f'OGNeuralODE, drift={drift_og:.2%}')
    ax.plot(t_eval, H_hnn, color='crimson', lw=1.2, label=f'HNNdopri5, drift={drift_hnn:.2%}')
    ax.plot(t_eval, H_lf, color='green', lw=1.2, label=f'HNNleapfrog, drift={drift_lf:.4%}')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('H(q,p)')
    ax.set_title(f' energy comparison on SHO for the four architectures, {TLONG}s rollout')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=.3)
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'comparison_sho.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"saved {fig_path}")

if __name__ == "__main__":
    main()