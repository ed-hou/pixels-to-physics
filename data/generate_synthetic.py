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
from src.systems.pendulum import SimplePendulum

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

    if sigma == 0.0:
        return q.copy(), p.copy()

    q_scale = np.max(np.abs(q))
    p_scale = np.max(np.abs(p))
    q_noisy = q + np.random.randn(*q.shape) * sigma *q_scale
    p_noisy = p + np.random.randn(*p.shape) * sigma * p_scale

    return q_noisy, p_noisy

def generate_sho_dataset(seed=69):
    """
    sho trajectories at varying initial amplitudes
    sampling from 0.5 to 2.5
    """
    rnjesus = np.random.RandomState(seed)
    sho = SimpleHarmonicOscillator(m=1.0, k=1.0)
    q0_samples = rnjesus.uniform(0.5, 2.5, N_TRAJECTORIES)

    #preallocate arrays
    all_t = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_q = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_p = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))
    all_H = np.zeros((N_TRAJECTORIES, N_POINTS_PER_TRAJ))

    for i, q0 in enumerate(q0_samples):
        t,q,p = sho.generate_trajectory(q0=q0, p0 = 0.0, t_span = (0, T_MAX), n_points = N_POINTS_PER_TRAJ)
        all_t[i] = t
        all_q[i] = q
        all_p[i] = p
        all_H[i] = sho.hamiltonian(q,p)

    return { 't': all_t, 'q': all_q, 'p': all_p, 'H': all_H, 'initial_conditions': np.column_stack([q0_samples, np.zeros_like(q0_samples)]), 'system_params': {'m': sho.m, 'k': sho.k}, 'system_name': 'SimpleHarmonicOscillator',}


def generate_pendulum_dataset(seed=67):
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


    return { 't': all_t, 'q': all_q, 'p': all_p, 'H': all_H,
             'initial_conditions': np.column_stack([q0_samples, np.zeros_like(q0_samples)]), 'system_params': {'m': pend.m, 'l': pend.l, 'g': pend.g},
             'system_name': 'SimplePendulum',}



def save_dataset(data, system_name, noise_level, seed=26):
        """
        Save a dataset to .npz with filename format: {system}_{noise_level}_{system_name}.npz

        """

        rnjesus = np.random.RandomState(seed+ int(noise_level*1000))
        #apply noise per trajectory

        q_noisy = np.zeros_like(data['q'])
        p_noisy = np.zeros_like(data['p'])
        for i in range(data['q'].shape[0]):
            #we use a different random state per trajectory to avoid corr. noise
            np.random.seed(seed+int(noise_level*6767) +i)
            q_noisy[i], p_noisy[i] = add_noise(data['q'][i], data['p'][i], noise_level)

        fname = f"{system_name}_noise{noise_level:.2f}.npz".replace('.', 'p', 1)
        fname = f"{system_name}_noise{int(noise_level*100):03d}.npz"
        fpath= os.path.join(OUTPUT_DIR, fname)

        np.savez(fpath, t= data['t'], q= q_noisy, p= p_noisy, q_clean=data['q'], p_clean=data['p'], H=data['H'], initial_conditions=data['initial_conditions'], noise_level=noise_level, system_name=data['system_name'],)
        print(f"  saved {fname} shape={q_noisy.shape} H_range= [{data['H'].min():.3f}], {data['H'].max():.3f}]")

def main():
    print(f"Generating SHO dataset({N_TRAJECTORIES} trajectories), {N_POINTS_PER_TRAJ} points each)")
    sho_data = generate_sho_dataset()
    for noise in NOISE_LEVELS:
        save_dataset(sho_data, 'sho', noise)

    print(f"\nGenerating pendulum dataset ({N_TRAJECTORIES} trajectories, {N_POINTS_PER_TRAJ} points each)")
    pend_data = generate_pendulum_dataset()
    for noise in NOISE_LEVELS:
        save_dataset(pend_data, 'pendulum', noise)

    print(f"\nAll data sved to {OUTPUT_DIR}")

    if __name__ == '__main__':
        main()


