import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.systems.sho import SimpleHarmonicOscillator
from src.models.hamiltonian_nn import HNN
from src.models.neural_ode import OGNeuralODE
from src.evaluation.metrics import (energy_drift, relativel2_error)

CHECKPOINT_HNN = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints', 'hamiltonianSHO.pt')
CHECKPOINT_OGNODE = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints', 'neural_ode_sho.pt')

TLONG=50.0
NPOINTS_LONG=5000
DEVICE = torch.device('cpu')
torch.set_default_dtype(torch.float32)

def load_hnn():
    model=HNN().to(DEVICE).to(torch.float32)
    model.load_state_dict(torch.load(CHECKPOINT_HNN, map_location=DEVICE))
    model.eval()
    return model

def load_ognode():
    model= OGNeuralODE().to(DEVICE).to(torch.float32)
    model.load_state_dict(torch.load(CHECKPOINT_OGNODE, map_location=DEVICE))
    model.eval()
    return model

def oracle_of_truth_trajectory(q0, p0, t_eval):
    sho=SimpleHarmonicOscillator(m=1.0, k=1.0)
    t,q,p = sho.generate_trajectory(q0=q0, p0=p0, t_span=(t_eval[0], t_eval[-1]), n_points=len(t_eval))
    H= sho.hamiltonian(q,p)
    return q, p, H


def hnn_rollout(model, q0, p0, t_eval):
    state0=torch.tensor([[q0,p0]], dtype=torch.float32).to(DEVICE)
    t_eval_t=torch.tensor(t_eval, dtype=torch.float32).to(DEVICE)
    traj=model.integrate(state0, t_eval_t)
    traj=traj.squeeze(1).detach().cpu().numpy()
    qpred, ppred=traj[:,0], traj[:,1]
    sho=SimpleHarmonicOscillator(m=1.0, k=1.0)
    Hpred=sho.hamiltonian(qpred,ppred)
    return qpred, ppred, Hpred

def ognode_rollout(model, q0, p0, t_eval):
    state0=torch.tensor([[q0,p0]], dtype=torch.float32).to(DEVICE)
    t_eval_t = torch.tensor(t_eval, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        traj=model.integrate(state0, t_eval_t)
    traj=traj.squeeze(1).detach().cpu().numpy()
    qpred, ppred= traj[:,0], traj[:,1]
    sho=SimpleHarmonicOscillator(m=1.0, k=1.0)
    Hpred=sho.hamiltonian(qpred,ppred)
    return qpred, ppred, Hpred

def main():
    hnn = load_hnn()
    ognode = load_ognode()
    print("loaded HNN and OGNeuralODE")

    q0, p0 = 1.0, 0.5
    sho = SimpleHarmonicOscillator(m=1.0, k=1.0)
    H0 = sho.hamiltonian(q0, p0)
    print(f"initial state: q0={q0}, p0={p0}, H0={H0}, H0={H0:.6f}")

    t_eval = np.linspace(0, TLONG, NPOINTS_LONG)
    qtrue, ptrue, Htrue = oracle_of_truth_trajectory(q0, p0, t_eval)
    qhnn, phnn, Hhnn = hnn_rollout(hnn, q0, p0, t_eval)
    qog, pog, Hog = ognode_rollout(ognode, q0, p0, t_eval)
    drift_hnn = energy_drift(Hhnn)
    drift_og = energy_drift(Hog)
    rel_err_2s_hnn = relativel2_error(np.column_stack([qhnn, phnn])[:int(2.0 / TLONG * NPOINTS_LONG)], np.column_stack([qtrue, ptrue])[:int(2.0 / TLONG * NPOINTS_LONG)])
    rel_err_full_hnn = relativel2_error(np.column_stack([qhnn, phnn]), np.column_stack([qtrue, ptrue]))
    print(f"hnn energy drift over {TLONG}s:        {drift_hnn:.4%}")
    print(f"OGNeuralODE energy drift on a scale of {TLONG}s: {drift_og:.4%}")
    print(f"hnn rel L2 error at 2s:  {rel_err_2s_hnn:.4%}")
    print(f"hnn rel L2 error at {TLONG}s: {rel_err_full_hnn:.4%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    horizons = [2.0, 10.0, TLONG]

    for ax, horizon in zip(axes, horizons):
        n = int(horizon / TLONG * NPOINTS_LONG)
        ax.plot(qtrue[:n], ptrue[:n], 'blue', lw=1.5, label='oracle of truth', alpha=.7)
        ax.plot(qhnn[:n], phnn[:n], 'orange', lw=1.0, label='HNN', alpha=.9)
        ax.scatter([q0], [p0], color='black', s=40, zorder=5, label='start')
        ax.set_xlabel('q')
        ax.set_ylabel('p')
        ax.set_title(f't from 0s to {horizon} secs')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(alpha=.3)
    plt.suptitle('HNN phase portraits at ever increasing horizons', y=1.02)
    plt.tight_layout()
    out1 = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'hnn_sho_phase_portraits.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')

    print(f"saved{out1}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(H0, color='blue', ls='--', lw=1, label=f'true H = {H0:.3f} (constant)')
    ax.plot(t_eval, Hog, 'gray', lw=1.0, alpha=.7, label=f'OGNeuralODE H(t) - {drift_og:.2%} drift')
    ax.plot(t_eval, Hhnn, 'crimson', lw=1.2, label=f'HNN H(t) - {drift_hnn:.2%} drift')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Hamiltonian H(q,p)')
    ax.set_title(f'energy drift over {TLONG}s rollout: HNN vs OGNeuralODE\nHNN max drift is {drift_hnn:.2%} of H(0), OGNeuralODE = {drift_og:.2%}')
    ax.legend()
    ax.grid(alpha=.3)
    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'hnn_sho_energydrift.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')

    print(f"saved{out2}")
    plt.show()
if __name__ == '__main__':
    main()
