"""generate synthetic training data for the sho and pendulum systems


for each system I
1)sample a range of initial conditions covering different energies
2) integrate each one to get a q,p trajectory, save trajectories and save noisy versions at variant noise levels
"""
import json
from datetime import datetime

import numpy as np
import os
import sys

# add project

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.systems.sho import SimpleHarmonicOscillator
from src.systems.pendulum import SimplePendulum

#config datasets
DATA_VERSION = '1.0'
N_TRAJECTORIES = 80
N_POINTS_PER_TRAJ = 1000
T_MAX = 10.0
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1]
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15
#SEEDS
SHO_SEED = 69
PENDULUM_SEED = 69
SHO_SPLIT_SEED = 1911
PENDULUM_SPLIT_SEED = 250
NOISE_SEED_BASE = 26


# directory where the .npz files will be saved

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'synthetic')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_noise (q,p, sigma):
    """
    add gaussian noise scaled by signal amplitude.
    sigma is fractional noise level of each variable's range mimicing the kind of noise one would get from imperfect measurement.
    """

    if sigma == 0.0:
        return q.copy(), p.copy()

    q_scale = np.max(np.abs(q))
    p_scale = np.max(np.abs(p))
    q_noisy = q + np.random.randn(*q.shape) * sigma *q_scale
    p_noisy = p + np.random.randn(*p.shape) * sigma * p_scale

    return q_noisy, p_noisy

def generate_sho_dataset(seed=SHO_SEED):
    """
    sho trajectories at varying initial amplitudes
    sampling from 0.5 to 2.5
    """
    rnjesus = np.random.RandomState(seed)
    sho = SimpleHarmonicOscillator(m=1.0, k=1.0)

    E_min, E_max = 0.1, 3.5

    initial_cond = []

    while len(initial_cond) < N_TRAJECTORIES:
        q0 = rnjesus.uniform(-2.5, 2.5)
        p0 = rnjesus.uniform(-2.5, 2.5)
        E = sho.hamiltonian(q0, p0)
        if E_min <= E <= E_max:
            initial_cond.append((q0, p0))
    initial_cond = np.array(initial_cond)

    #preallocate arrays
    all_t = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_q = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_p = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_H = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))

    for i, (q0, p0) in enumerate(initial_cond):
        t,q,p = sho.generate_trajectory(q0=q0, p0=p0, t_span = (0, T_MAX), n_points = N_POINTS_PER_TRAJ)
        all_t[i] = t
        all_q[i] = q
        all_p[i] = p
        all_H[i] = sho.hamiltonian(q,p)

    return { 't': all_t, 'q': all_q, 'p': all_p, 'H': all_H, 'initial_cond': initial_cond, 'system_params': {'m': sho.m, 'k': sho.k}, 'system_name': 'SimpleHarmonicOscillator',}


def generate_pendulum_dataset(seed=PENDULUM_SEED):
    """
    I sample q0 from 0.3 to 1.5rad to demonstrate nonlinearity effects. stay in libration regime
    """

    rnjesus = np.random.RandomState(seed)
    pend = SimplePendulum(m=1.0, l=1.0, g=9.81)

    E_min, E_max = 0.5, 9.0
    initial_cond = []
    while len(initial_cond) < N_TRAJECTORIES:
        q0 = rnjesus.uniform(-1.6, 1.6)
        p0 = rnjesus.uniform(-3.0, 3.0)
        E = pend.hamiltonian(q0, p0)
        if E_min <= E <= E_max:
            initial_cond.append((q0, p0))
    initial_cond = np.array(initial_cond)

    all_t = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_q = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_p = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_H = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))

    for i, (q0, p0) in enumerate(initial_cond):
        t,q,p = pend.generate_trajectory(q0=q0, p0=p0, t_span=(0, T_MAX), n_points=N_POINTS_PER_TRAJ)
        all_t[i] = t
        all_q[i] = q
        all_p[i] = p
        all_H[i] = pend.hamiltonian(q,p)


    return { 't': all_t, 'q': all_q, 'p': all_p, 'H': all_H,
             'initial_cond': initial_cond, 'system_params': {'m': pend.m, 'l': pend.l, 'g': pend.g},
             'system_name': 'SimplePendulum',}

def make_split_helper(n_trajectories, seed = 67):
    """
    split trajectory indices into train,val,test sets. I decide to split by trajectory instead of time for info integrity
    """

    rnjesus = np.random.RandomState(seed)
    indices = np.arange(n_trajectories)
    rnjesus.shuffle(indices)
    n_train = int(TRAIN_FRAC * n_trajectories)
    n_val = int(VAL_FRAC * n_trajectories)

    return { 'train': indices[:n_train],
             'val': indices[n_train:n_train+n_val],
             'test': indices[n_train+n_val:]}

def save_dataset(data, system_name, noise_level, splits, seed=NOISE_SEED_BASE):
        """
        Save dataset with noise and traj split assignments

        """


        #apply noise per trajectory

        q_noisy = np.zeros_like(data['q'])
        p_noisy = np.zeros_like(data['p'])
        for i in range(data['q'].shape[0]):
            #we use a different random state per trajectory to avoid corr. noise
            np.random.seed(seed+int(noise_level*6767) +i)
            q_noisy[i], p_noisy[i] = add_noise(data['q'][i], data['p'][i], noise_level)


        fname = f"{system_name}_noise{int(noise_level*100):03d}.npz"
        fpath= os.path.join(OUTPUT_DIR, fname)

        np.savez(fpath, t= data['t'], q= q_noisy, p= p_noisy, q_clean=data['q'], p_clean=data['p'], H=data['H'], initial_cond=data['initial_cond'], train_idx=splits['train'], val_idx=splits['val'],
        test_idx=splits['test'], noise_level=noise_level, system_name=data['system_name'],)

        print(f"  saved {fname}  shape={q_noisy.shape}  "
          f"split={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}  "
          f"H=[{data['H'].min():.2f}, {data['H'].max():.2f}]")

def main():
    print(f"Generating SHO dataset({N_TRAJECTORIES} trajectories)")
    sho_data = generate_sho_dataset()
    sho_splits = make_split_helper(N_TRAJECTORIES, seed=SHO_SPLIT_SEED)
    for noise in NOISE_LEVELS:
        save_dataset(sho_data, 'sho', noise, sho_splits)

    print(f"\nGenerating pendulum dataset ({N_TRAJECTORIES} trajectories, {N_POINTS_PER_TRAJ} points each)")
    pend_data = generate_pendulum_dataset()
    pend_splits = make_split_helper(N_TRAJECTORIES, seed = PENDULUM_SPLIT_SEED)
    for noise in NOISE_LEVELS:
        save_dataset(pend_data, 'pendulum', noise, pend_splits)

    metadata = {
        'data version': DATA_VERSION,
        'generated_at': datetime.now().isoformat(),
        'n_trajectories': N_TRAJECTORIES,
        'n_points_per_trajectory': N_POINTS_PER_TRAJ,
        't_max': T_MAX,
        'noise levels': NOISE_LEVELS,
        'split_fractions': {'train': TRAIN_FRAC, 'val': VAL_FRAC, 'test': round(1- TRAIN_FRAC-VAL_FRAC, 2)},
        'sampling_strategy': '2D phase space sampling with energy band rejection', 'sho_energy_band': [0.1, 3.5],
        'pendulum_energy_band': [0.5, 9.0], 'integrator': 'DOP853 with rtol=atol=1e-12', 'seeds': {'sho': SHO_SEED, 'pendulum': PENDULUM_SEED, 'sho_split': SHO_SPLIT_SEED, 'pendulum_split': PENDULUM_SPLIT_SEED, 'noise_base': NOISE_SEED_BASE},

    }
    metadata_path = os.path.join(OUTPUT_DIR, 'dataset_info.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset metadata written to {metadata_path}")
    print(f"All data saved to {OUTPUT_DIR}")


if __name__ == '__main__':
        main()


