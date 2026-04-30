"""generate synthetic training data for the sho and pendulum systems


for each system I
1)sample a range of initial conditions covering different energies
2) integrate each one to get a q,p trajectory, save trajectories and save noisy versions at variant noise levels
"""

import numpy as np
import os
import sys

# add project

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.systems.sho import SimpleHarmonicOscillator
from src.systemspendulum import SimplePendulum

#config datasets

N_TRAJECTORIES = 50
N_POINTS_PER_TRAJ = 1000
T_MAX = 10.0
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]

# directory where the .npz files will be saved

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'synthetic')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_noise (q,p, sigma):
    """
    add gaussian noise scaled by signal amplitude.
    sigma is fractional noise level of each variable's range mimicing the kind of noise one would get from imperfect measurement.
    """

    if sigma == 0:
        return q.copy(), p.copy()

    q_scale = np.max(np.abs(q))
    p_scale = np.max(np.abs(p))

    q_noisy = q + np.random.randn(*q.shape) * sigma *q_scale
    p_noisy = p + np.random.randn(*p.shape) * sigma * p_scale

    return q_noisy, p_noisy

def generate_sho_dataset(seed=26):
    """
    I sample q0 from 0.3 to 1.5rad to demonstrate nonlinearity effects. stay in libration regime
    """

    rnjesus = np.random.RandomState(seed)
    pend = SimplePendulum(m=1.0, l=1.0, g=9.81)
    q0_samples = rnjesus.uniform(0.3, 1.5, N_TRAJECTORIES)

    all_t = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_q = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_p = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_H = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))

    for i, q0 in enumerate(q0_samples):
        t,q,p = pend.generateTrajectory(q0=q0, p0=0.0, t_span=(0, T_MAX), n_points=N_POINTS_PER_TRAJ)
        all_t[i] = t
        all_q[i] = q
        all_p[i] = p
        all_H[i] = pend.hamiltonian(q,p)

    return {
        't': all_t,
        'q': all_q,
        'p': all_p,
        'H': all_H,
        'initial_conditions': np.column_stack([q0_samples, np.zeros_like(q0_samples)]),
        'system_params': {'m:' pend.m, 'l:' pend.l, 'g:' pend.g},
        'system_name': 'SimplePendulum',

    }