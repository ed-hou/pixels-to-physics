"""
HNN trainer for simpleharmonicoscillator. The same thing for train_node for loading data, windowingetc
cpu pinned and same window=10 single step endpoint regression pattern
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import torch
import torch.nn as nn
from src.models.hamiltonian_nn import HNN

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'synthetic', 'sho_noise000.npz')
N_EPOCHS = 400
LR_MAX = 1e-3
LR_MIN = 1e-5
WARMUP=500
GRADCLIP=1.0
BATCHSIZE = 256
CURRICULUM = [(0,2), (50,5), (150,10)]
DEVICE=torch.device('cpu')
print(f"using device: {DEVICE}")

def load_sho_train_data():
    data=np.load(DATA_PATH)
    train_idx= data['train_idx']
    t=data['t'][train_idx]
    q=data['q_clean'][train_idx]
    p=data['p_clean'][train_idx]
    return t,q,p

def make_windowpairs(t,q,p,window):
    ntraj,T= q.shape
    starts, dts, targets = [],[],[]
    for i in range(ntraj):
        for j in range(T-window):
            start = np.array([q[i,j], p[i,j]])
            target = np.array([q[i,j+window], p[i,j+window]])
            dt=t[i, j+window] - t[i,j]
            starts.append(start)
            dts.append(dt)
            targets.append(target)
    return (torch.tensor(np.array(starts), dtype=torch.float32), torch.tensor(np.array(dts), dtype=torch.float32), torch.tensor(np.array(targets),dtype=torch.float32))

def current_window(epoch):
    #walk curriculum table backwards and return the most recent active window
    mostrecent_window = CURRICULUM[0][1]
    for start_epoch, i in CURRICULUM:
        if epoch >= start_epoch:
            mostrecent_window = i
    return mostrecent_window

def warmup_lr(step):
    #warmup then constant lr_max

    if step>= WARMUP:
        return LR_MAX
    return (LR_MIN+(LR_MAX-LR_MIN)*(step/WARMUP))

def train():
    t,q, p = load_sho_train_data()
    print(f"Loaded {q.shape[0]} training trajectories with length {q.shape[1]}")
    #comp nomrmalization stats
    state_mean=np.array([q.mean(), p.mean()], dtype=np.float32)
    state_std=np.array([q.std(), p.std()], dtype=np.float32)
    print(f"state mean: q={state_mean[0]:.4f}, p={state_mean[1]:.4f}")
    print(f"state std: q= {state_std[0]:.4f}, p= {state_std[1]:.4f}")
    torch.set_default_dtype(torch.float32)
    model=HNN().to(DEVICE).to(torch.float32)
    model.func.set_normalization(state_mean, state_std)
    optim=torch.optim.Adam(model.parameters(), lr=LR_MIN)
    lossfunction = nn.MSELoss()

    #now build window pairs
    window_now =-1
    starts = dts=targets=None
    nsamples=0
    t_eval=None
    global_step=0
    for epoch in range(N_EPOCHS):
        mostrecent_window=current_window(epoch)
        if mostrecent_window!= window_now:
            #rebuild dataset at the new window
            print(f"epoch {epoch:3d}: curriculum jump to WINDOW={mostrecent_window}, rebuilding windowpairs")
            starts, dts, targets = make_windowpairs(t,q,p, mostrecent_window)
            print(f"  made {starts.shape[0]} window pairs")
            dt=dts[0].item()
            assert torch.allclose(dts, dts[0]), "expected uniform dt for all pairs"
            t_eval = torch.tensor([0.0, dt], dtype=torch.float32).to(DEVICE)
            starts=starts.to(DEVICE)
            targets=targets.to(DEVICE)
            nsamples=starts.shape[0]
            window_now=mostrecent_window

        perm=torch.randperm(nsamples)
        epoch_loss=0.0
        nbatches=0

        for i in range(0, nsamples, BATCHSIZE):
            idx=perm[i:i+BATCHSIZE]
            batch_start=starts[idx]
            batch_target=targets[idx]
            #set LR per warmup schedule
            for g in optim.param_groups:
                g['lr']=warmup_lr(global_step)

            traj=model.integrate(batch_start, t_eval)
            pred= traj[-1]
            loss=lossfunction(pred, batch_target)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADCLIP)
            optim.step()
            epoch_loss+=loss.item()

            nbatches +=1
            global_step+=1
        if epoch%10==0 or epoch == N_EPOCHS-1:
            lr_current= optim.param_groups[0]['lr']
            print(f"epoch {epoch:3d}, win {mostrecent_window}, lr {lr_current:.2e}, loss {epoch_loss/nbatches:.6e}")

    out_dir=os.path.join(os.path.dirname(__file__), '..', '..', 'recents', 'checkpoints')
    os.makedirs(out_dir, exist_ok=True)
    out_path=os.path.join(out_dir, 'hamiltonianSHO.pt')
    torch.save(model.state_dict(), out_path)
    print(f"saved to {out_path}")

if __name__=='__main__':
    train()



