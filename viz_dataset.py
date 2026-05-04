import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = 'data/synthetic'
FIG_DIR = 'results/figures'
os.makedirs(FIG_DIR, exist_ok=True)

def plot_dataset(system_name, noise_level=0.05):
    """generate 4 panels for the system"""
    fname = f"{system_name}_noise{int(noise_level * 100):03d}.npz"
    data = np.load(os.path.join(DATA_DIR, fname))

    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']
    q_clean = data['q_clean']
    p_clean = data['p_clean']
    q_noisy = data['q']
    p_noisy = data['p']
    H = data['H']
    ics = data['initial_cond']

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(f"{system_name.upper()} dataset — noise={noise_level}", fontsize=14)

    # panel 1: phase portrait colored by split
    ax = axes[0, 0]
    for i in train_idx:
        ax.plot(q_clean[i], p_clean[i], color='steelblue', alpha=0.4, linewidth=0.6)
    for i in val_idx:
        ax.plot(q_clean[i], p_clean[i], color='orange', alpha=0.6, linewidth=0.8)
    for i in test_idx:
        ax.plot(q_clean[i], p_clean[i], color='crimson', alpha=0.6, linewidth=0.8)
    # legend hack — plot one invisible line per split for the legend
    ax.plot([], [], color='steelblue', label=f'train ({len(train_idx)})')
    ax.plot([], [], color='orange', label=f'val ({len(val_idx)})')
    ax.plot([], [], color='crimson', label=f'test ({len(test_idx)})')
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_title('Phase portrait by split')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')


    # energy distribution panel
    ax = axes[0, 1]
    energies_per_traj = H[:, 0]
    ax.hist(energies_per_traj, bins=20, color='slategray', edgecolor='black')
    ax.set_xlabel('total energy H')
    ax.set_ylabel('# trajectories')
    ax.set_title(f'Energy distribution (range: {H.min():.2f} to {H.max():.2f})')

    # scatter of init cond
    ax = axes[1, 0]
    ax.scatter(ics[train_idx, 0], ics[train_idx, 1], color='steelblue', alpha=0.6, label='train', s=30)
    ax.scatter(ics[val_idx, 0], ics[val_idx, 1], color='orange', alpha=0.8, label='val', s=30)
    ax.scatter(ics[test_idx, 0], ics[test_idx, 1], color='crimson', alpha=0.8, label='test', s=30)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.set_xlabel('q0')
    ax.set_ylabel('p0')
    ax.set_title('InitCond of Phase space')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')

    # comparison of noise for single trajectory
    ax = axes[1, 1]
    i = train_idx[0]
    t = data['t'][i]
    ax.plot(t, q_clean[i], label='q clean', linewidth=2, color='steelblue')
    ax.plot(t, q_noisy[i], label='q noisy', alpha=0.6, linewidth=0.8, color='steelblue')
    ax.plot(t, p_clean[i], label='p clean', linewidth=2, color='crimson')
    ax.plot(t, p_noisy[i], label='p noisy', alpha=0.6, linewidth=0.8, color='crimson')
    ax.set_xlabel('t')
    ax.set_title(f'Trajectory #{i}: clean versus noisy comparison')
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, f'dataset_preview_{system_name}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"saved {out_path}")

def main():
        for system in ['sho', 'pendulum']:
            plot_dataset(system, noise_level=0.05)
        print ("\ndone. check results/figures/")

if __name__ == "__main__":
        main()