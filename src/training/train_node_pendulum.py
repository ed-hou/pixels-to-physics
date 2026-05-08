"""
gonna have short windows where w=10 which will give stable gradients and relatively fast convergence
trains the ogneural ode on pendulum data
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
#torch.set_default_dtype(torch.float32)
import torch.nn as nn
from src.models.neural_ode import OGNeuralODE

DATA_PATH= os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'synthetic', 'pendulum_noise000.npz')
WINDOW=10
N_EPOCHS= 200
LR= 1e-3
BATCH_SIZE =256

"""
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("using device: mps (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("using device: cuda")
else:
    DEVICE = torch.device('cpu')
    print("using device: cpu")
"""
DEVICE = torch.device('cpu')
print(f"using device: {DEVICE}")


def load_pendulum_train_data():
    #load training trajectories from synth dataset
    data= np.load(DATA_PATH)
    train_idx=data['train_idx']
    t=data['t'][train_idx]
    q=data['q_clean'][train_idx]
    p=data['p_clean'][train_idx]
    return t,q,p

def make_windowpairs(t,q,p, window):
    #this will slice trajectories into (start_state, dt_vec, target_state) tuples. Each trajectory of length T yields T-window training pairs.
    n_traj, T=q.shape
    starts, dts, targets = [], [], []
    for i in range(n_traj):
        for j in range(T-window):
            start=np.array([q[i, j], p[i, j]])
            target = np.array([q[i,j+window], p[i, j+window]])
            dt= t[i, j+window] - t[i,j]
            starts.append(start)
            dts.append(dt)
            targets.append(target)

    return (torch.tensor(np.array(starts), dtype=torch.float32), torch.tensor(np.array(dts), dtype=torch.float32), torch.tensor(np.array(targets), dtype=torch.float32))

def train():
    t,q,p = load_pendulum_train_data()
    print(f"loaded {q.shape[0]} training trajectories of length {q.shape[1]}")
    starts, dts, targets = make_windowpairs(t,q,p, WINDOW)
    print(f"made{starts.shape[0]} window pairs")
    dt = dts[0].item()
    assert torch.allclose(dts, dts[0]), "expected uniform dt across pairs"
    t_eval = torch.tensor([0.0, dt], dtype=torch.float32).to(DEVICE)
    torch.set_default_dtype(torch.float32)
    model = OGNeuralODE().to(DEVICE).to(torch.float32)
    optim = torch.optim.Adam(model.parameters(), lr = LR)
    loss_fn = nn.MSELoss()

    starts = starts.to(DEVICE)
    targets = targets.to(DEVICE)
    nsamples=starts.shape[0]

    for epoch in range(N_EPOCHS):
        perm=torch.randperm(nsamples)
        epoch_loss=0.0
        nbatches=0

        for i in range(0, nsamples, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            batch_start = starts[idx]
            batch_target = targets[idx]

            traj= model.integrate(batch_start, t_eval)
            pred = traj[-1]
            loss = loss_fn(pred, batch_target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            nbatches+=1

        if epoch %10 == 0 or epoch == N_EPOCHS-1:
            print(f"epoch {epoch:3d}, loss {epoch_loss/nbatches:.6e}")
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'checkpoints')

    os.makedirs(out_dir, exist_ok=True)
    out_path=os.path.join(out_dir, 'neural_ode_pendulum.pt')
    torch.save(model.state_dict(), out_path)
    print(f"saved to {out_path}")

if __name__ == "__main__":
    train()

