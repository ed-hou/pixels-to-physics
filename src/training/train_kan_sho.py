"""
kan trainer for sho
"""

import os
import numpy as np
import torch
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))



from src.systems.sho import SimpleHarmonicOscillator
from src.models.kan_wrapper import HamiltonianKAN

data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'synthetic', 'sho_noise000.npz')
steps=50
lamb=.001
lamb_entropy=2.0
DEVICE = torch.device('cpu')


def load_sho_train_data():
    data=np.load(data_path)
    train_idx=data['train_idx']
    q=data['q_clean'][train_idx]
    p=data['p_clean'][train_idx]

    return q,p
def train():
    q,p=load_sho_train_data()
    print(f"loaded {q.shape[0]} training trajectories w/length {q.shape[1]}")
    sho=SimpleHarmonicOscillator(m=1.0, k=1.0)
    hkan=HamiltonianKAN(sho)
    dataset=hkan.build_dataset(q,p)

    print("dataset keys:", list(dataset.keys()))
    print("sample shapes:", {k: getattr(v, 'shape', type(v)) for k, v in dataset.items()})
    print(f"built dataset w/{dataset['train_input'].shape[0]} train+ {dataset['test_input'].shape[0]} test points")

    results=hkan.model.fit(dataset,lamb=lamb,lamb_entropy=lamb_entropy)

    out_dir=os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'checkpoints')
    os.makedirs(out_dir, exist_ok=True)
    out_path=os.path.join(out_dir, 'kanSHO.pt')
    torch.save(hkan.model.state_dict(), out_path)
    print(f"saved to {out_path}")

if __name__=='__main__':
    train()