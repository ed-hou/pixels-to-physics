"""
This will be a diagnostic for the ogneuralode on long rolllouts.

We trained on short windows at 10 steps and .1s integration so now we will extend to 50 seconds, at 5x longer than any training trajectory
Expect longer rollouts to drift considerably and will observe h(t) that will deviate from the constant
"""

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.systems.sho import SimpleHarmonicOscillator
from src.models.neural_ode import OGNeuralODE
from src.evaluation.metrics import energy_drift, relativel2_error

CHECKPOINTS = os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints', 'neural_ode_sho.pt')
T_LONG = 50.0
N_POINTS_LONG = 5000
DEVICE = torch.device('cpu')
torch.set_default_dtype(torch.float32)

def load_trained_model():
    model = OGNeuralODE().to(DEVICE).to(torch.float32)
    model.load_state_dict(torch.load(CHECKPOINTS, map_location=DEVICE))
    model.eval()
    return model

def oracle_of_truth_trajectory(q0, p0, t_eval):
    #generate the true sho trajectory for comparison
    sho = SimpleHarmonicOscillator(m=1.0, k=1.0)
    t, q,p = sho.generate_trajectory (q0=q0, p0=p0, t_span=(t_eval[0], t_eval[-1]), n_points=len(t_eval))
    H = sho.hamiltonian(q,p)
    return q, p, H

def ogneuralode_rollout(model, q0, p0, t_eval):
    #run the neural ode over t_eval
    state0 = torch.tensor([[q0, p0]], dtype=torch.float32).to(DEVICE)
    t_eval_t= torch.tensor(t_eval, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        traj=model.integrate(state0, t_eval_t)
    traj = traj.squeeze(1). cpu(). numpy()
    qpred, ppred = traj[:, 0], traj[:, 1]

    #compute H with true hamiltonian on predicted states
    sho = SimpleHarmonicOscillator(m=1.0, k=1.0)
    H_pred = sho.hamiltonian(qpred, ppred)
    return qpred, ppred, H_pred

def main():
    model =load_trained_model()
    print("loaded trained neural ode")
    #initial condition
    q0, p0 = 1.0, .5
    sho = SimpleHarmonicOscillator(m=1.0, k=1.0)
    H0 = sho.hamiltonian(q0, p0)
    print(f"initial state: q0= {q0}, p0 = {p0}, H0 = {H0}, H0={H0:.6f}")

    #run w/longrollout
    t_eval= np.linspace(0, T_LONG, N_POINTS_LONG)
    qtrue, ptrue, Htrue = oracle_of_truth_trajectory(q0, p0, t_eval)
    qpred, ppred, Hpred = ogneuralode_rollout(model, q0, p0, t_eval)

    #quant summary
    drift = energy_drift(Hpred)
    rel_err_2s = relativel2_error( np.column_stack([qpred, ppred])[:int(2.0 / T_LONG * N_POINTS_LONG)], np.column_stack([qtrue, ptrue])[:int(2.0 / T_LONG * N_POINTS_LONG)])
    rel_err_full = relativel2_error(np.column_stack([qpred, ppred]), np.column_stack([qtrue, ptrue]))

    print(f"energy drift over {T_LONG}s: {drift:.4%}")
    print(f"relative L2 error at 2s:  {rel_err_2s:.4%}")
    print(f"relative L2 error at {T_LONG}s: {rel_err_full:.4%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    horizons = [2.0, 10.0, T_LONG]
    for ax, horizon in zip(axes, horizons):
        n = int(horizon / T_LONG *N_POINTS_LONG)
        ax.plot(qtrue[:n], ptrue[:n], 'blue', lw = 1.5, label= 'oracle of truth', alpha = .7)
        ax.plot(qpred[:n], ppred[:n], 'orange', lw =1.0 , label = 'Neural ode', alpha=.9)
        ax.scatter([q0], [p0], color='black', s=40, zorder=5, label= 'start')
        ax.set_xlabel('q')
        ax.set_ylabel('p')
        ax.set_title(f't from 0 seconds to {horizon} seconds')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(alpha=.3)
    plt.suptitle('OGNEURALODE phase portraits at ever increasing horizons', y=1.02)
    plt.tight_layout()
    out1= os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'neural_ode_phase_portraits.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"saved{out1}")

    #figure 2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(H0, color = 'blue', ls = '--', lw=1, label= f'true H = {H0:.3f} (constant)')
    ax.plot(t_eval, Hpred, 'crimson', lw=1.2, label ='Neural ODE for H(t)')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Hamiltonian H(q,p)')
    ax.set_title(f'Energy drift over {T_LONG} rollout for og neural ode\n' f' max drift = {drift:.2%}of H(0)')
    ax.legend()
    ax.grid(alpha=.3)
    plt.tight_layout()
    out2=os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'neural_ode_energydrift.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"saved{out2}")
    plt.show()

if __name__ == "__main__":
     main()





