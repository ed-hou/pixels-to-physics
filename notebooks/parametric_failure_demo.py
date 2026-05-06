"""
I am guessing wrong here on purpose. If we have zero cognizance of the functional form of the hamiltonian, our parametric fitting will force us to guess.
We will the the SHO form, a*p^2 + b*q^2 to pendulum data and observe the residuals.

We're gonna compare this to the results of KAN.
"""

import os
import sys

from PIL.FitsImagePlugin import FitsImageFile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from src.systems.pendulum import SimplePendulum
from src.models.parametric_calibration import ParametricHamiltonianFit

#now I generate pendulum data at large amps so the nonlinearity is nontrivial, where at small angles sin(q)~q and pendulum will look like SHO. the wrong from would work so we want q0 large enough that (q-cosq) deviates meaningfully from q^2/2
pend = SimplePendulum(m=1.0, l=1.0, g=9.81)
q0= 1.5 #this is around 85deg which is well into nonlinear regime
t,q,p = pend.generate_trajectory(q0=q0, p0=0.0, t_span=(0,10), n_points=2000)
H_true = pend.hamiltonian(q,p)
print(f"Pendulum at q0={q0} rad ({np.degrees(q0): .1f} deg)")
print(f" q range: [{q.min():.3f}, {q.max():.3f}")
print(f" H={H_true[0]:.4f} (constant along trajectory)")
print()
#we fit both forms to the same data
fitter_correct = ParametricHamiltonianFit()
correct = fitter_correct.fit_pendulum(q,p,H_true)

fitter_wrong = ParametricHamiltonianFit()
wrong=fitter_wrong.fit_sho(q,p, H_true)

print(f"Correct form (pendulum): {correct['formula']}")
print(f" SSE = {np.sum(fitter_correct.residuals**2):.4d}")
print(f" max |residual| = {np.max(np.abs(fitter_correct.residuals)):.4e}")
print()
print(f"Wrong form (SHO):    {wrong['formula']}")
print(f" SSE= {np.sum(fitter_wrong.residuals**2):.4e}")
print(f"    max |residual| = {np.max(np.abs(fitter_wrong.residuals)):.4e}")
print()
print(f"SSE ratio (wrong/correct): {np.sum(fitter_wrong.residuals**2)/ np.sum(fitter_correct.residuals**2):.2e}")

"""
Draw figure with residuals vs q for both fits in sidebyside view
-correct fit residuals should have a flat line where 0
-the wrong fit's residuals should show soemthing as a function of q
"""

fig, axes = plt.subplots(1,2, figsize=(12,4.5), sharey=False)
axes[0].scatter(q, fitter_correct.residuals, s=3, alpha=0.5, color='steelblue')
axes[0].axhline()

