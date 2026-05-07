import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.systems.sho import SimpleHarmonicOscillator

sho = SimpleHarmonicOscillator(m=1.0, k=1.0)
fig, axes =plt.subplots(1, 3, figsize = (15, 5))

t, q, p = sho.generate_trajectory(q0 = 1.0, p0 = 0.0, t_span=(0, 20), n_points = 1000)

axes[0].plot(t, q, label = 'q (position)')
axes[0].plot(t, p, label = 'p (momentum)')
axes[0].set_xlabel('t')
axes[0].set_title('SHO Time Series')
axes[0].legend()

#plot 2 where we have q(t) and p(t) showin the time series

for q0 in [0.5, 1.0, 1.5, 2.0]:
    t, q, p = sho.generate_trajectory(q0 = q0, p0 = 0.0, t_span=(0, 10), n_points = 500)
    axes[1].plot(q, p, label = f'E={sho.hamiltonian(q0, 0.0):.2f}')
axes[1].set_xlabel('q')
axes[1].set_ylabel('p')
axes[1].set_title('Phase Outlook')
axes[1].set_aspect('equal')
axes[1].legend()

#plot 3: energy conservation here H(t) should be flat

t, q, p = sho.generate_trajectory(q0 = 1.0, p0 = 0.0, t_span=(0, 50), n_points = 5000)
H = sho.hamiltonian(q,p)
axes[2].plot(t, H-H[0])
axes[2].set_xlabel('t')
axes[2].set_ylabel('H(t) - H(0)')
axes[2].set_title('Energy Drift')
axes[2].ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0, 0) )
plt.tight_layout()
plt.savefig('results/figures/sho_vertical.png', dpi = 150)
plt.show()
print("I saved the this plot to  results/figure/sho_verification.png")



