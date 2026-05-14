import os
import sys

from sympy.logic.inference import pl_true

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.systems.pendulum import SimplePendulum
from src.models.kan_wrapper import HamiltonianKAN


data_path=os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', 'pendulum_noise000.npz')
checkpoint_path=os.path.join(os.path.dirname(__file__), '..', 'results', 'checkpoints', 'kanPendulum.pt')

steps_refit=50
lib=['x', 'x^2', 'cos']

def load_pend_data():
    data=np.load(data_path)
    train_idx=data['train_idx']
    q=data['q_clean'][train_idx]
    p=data['p_clean'][train_idx]
    return q, p

def main():
    pend=SimplePendulum(m=1.0, k=1.0, g=9.81)
    hkan=HamiltonianKAN(pend) #dataset for refit after pruning
    q,p=load_pend_data()
    dataset=hkan.build_dataset(q,p)

    hkan_loaded = HamiltonianKAN(pend)
    hkan_loaded.model.load_state_dict(torch.load(checkpoint_path))
    model = hkan_loaded.model
    print(f"loaded kan from {checkpoint_path}")
    model.get_act(dataset['train_input'])
    #plotting the preprune diagram so we can visualize what specifically gets dropped
    plt.figure(figsize=(8, 8))
    model.plot()
    out0 = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'kan_pendulum_preprune.png')
    plt.savefig(out0, dpi=150, bbox_inches='tight')
    print(f"saved {out0}")

    model = model.prune()
    print(f"pruned. refitting on the smaller graph")
    model.fit(dataset, opt='LBFGS', steps=steps_refit)
    plt.figure(figsize=(8, 8))
    model.plot()
    out1 = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'kan_pendulum_pruned.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"saved {out1}")

    #gonna pick x^2 on p and q edges as sho contains x and x^2
    model.auto_symbolic(lib=lib)
    formula=model.symbolic_formula()[0][0]
    #1,0,0 picked x^2 with r2=.999 but it is actually linear pass through
    model.fix_symbolic(1,0,0,'x')
    model.fit(dataset, opt='LBFGS', steps=50)
    formula=model.symbolic_formula()[0][0]
    print(f"discovered formula: H= {formula}")

    qg= np.linspace(-1.5, 1.5, 50)
    pg=np.linspace(-2., 2., 50)
    Q,P = np.meshgrid(qg, pg)
    H_true = pend.hamiltonian(Q.flatten(), P.flatten())
    test_in = torch.tensor(np.stack([Q.flatten(), P.flatten()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        H_pred=model(test_in).squeeze().numpy()
    rel_error= np.abs(H_pred-H_true).mean() /np.abs(H_true).mean()
    print(f"mean rel error on phase space rid: {rel_error:.4%}")
    print(f"true H=.5*q^2+ 9.81*(1-cos(q))")

    out2 = os.path.join(os.path.dirname(__file__), '..', 'results', 'tables', 'kan_pendulum_symbolic.txt')
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    with open(out2, 'w') as f:
        f. write(f"discovered fomula:\n H= {formula}\n\n")
        f.write(f"true formula:\n =.5*p^2+9.81*(1-cos(q))\n\n")
        f.write(f"mean rel error on phase space grid: {rel_error:.4%} \n")

    print(f"saved{out2}")

if __name__=='__main__':
    main()
