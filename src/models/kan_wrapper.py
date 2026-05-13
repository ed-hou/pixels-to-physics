from kan import KAN

import torch

import numpy as np

class HamiltonianKAN:
    #2,5,5,1 with 2 inputs at (q,p), grid=5, k=3, seed=1
    def __init__(self, system, width=[2,5,5,1], grid=5, k=3, seed=1):
        self.system=system
        self.width= width
        self.model=KAN(width=width, grid=grid, k=k, seed=seed, device='cpu')

    def build_dataset(self,q,p):
        """
        flatten (n_traj, T) trajectories into pointwise(N, 2) inputs with
        H_true labels.

        """

        q_flat = q.flatten().astype(np.float32)
        p_flat=p.flatten().astype(np.float32)
        H_flat=self.system.hamiltonian(q_flat, p_flat).astype(np.float32)
        inputs=torch.from_numpy(np.stack([q_flat,p_flat], axis=1))
        labels=torch.from_numpy(H_flat).unsqueeze(-1)

        n=inputs.shape[0]
        rng=torch.Generator().manual_seed(1)
        perm=torch.randperm(n, generator=rng)
        ntrain=int(.8*n)
        return {'train_input':inputs[perm[:ntrain]], 'train_label': labels [perm[:ntrain]], 'test_input': inputs[perm[ntrain:]], 'test_label': labels [perm[ntrain:]]}




