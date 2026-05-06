"""
testing for four properties:
1) observe fit_sho recovers the actual coefficients on sho data
2) same with fit_pendulum
3) fit_unknown correctly ranks the pendulum form above the sho form on pendulum data
4) wrong form failre: fitting H=a*p^2+b*q^2 to pendulum data produces residuals that are catastrophically larger than the correct form's residuals.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.systems.sho import SimpleHarmonicOscillator
from src.systems.pendulum import SimplePendulum
from src.models.parametric_calibration import ParametricHamiltonianFit

def test_sho_recovery():
    #clean data should yield a=b=.5
    sho = SimpleHarmonicOscillator(m=1.0, k=1.0)
    t,q,p= sho.generate_trajectory(q0=1.0, p0=.5, t_span=(0,10), n_points=2000)
    H_true=sho.hamiltonian(q,p)
    fitter= ParametricHamiltonianFit()
    result= fitter.fit_sho(q,p,H_true)

    print(f"fit_sho on SHO data: a= {result['a']:.8f}, b= {result['b']:.8f}")
    print(f"   max residual: {np.max(np.abs(fitter.residuals)):.2e}")
    assert abs(result['a'] - .5) < 1e-6, f"a recovery failed: {result['a']}"
    assert abs(result['b'] - .5) < 1e-6, f"b recovery failed: {result['b']}"
    print("PASSED: SHO coefficients recovered")

def test_pendulum_recovery():
    ###clean pendulum data should yield a=.5, b=9.81 to high precision
    pend = SimplePendulum(m=1.0, l=1.0, g=9.81)
    t,q,p = pend.generate_trajectory(q0=1.0, p0=0.0, t_span=(0,10), n_points=2000)
    H_true=pend.hamiltonian(q,p)
    fitter= ParametricHamiltonianFit()
    result=fitter.fit_pendulum(q,p,H_true)

    print(f"fit_pendulum on pendulum data: a = {result['a']:.8f}, b = {result['b']:.8f}")
    print(f"  max residual: {np.max(np.abs(fitter.residuals)):.2e}")
    #b should recover mgl=1.0*9.81*1,0=9.81
    assert abs(result['a']-.5) < 1e-6, f"a recovery failed: {result['a']}"
    assert abs(result['b']-9.81) < 1e-5, f"b recovery failed: {result['b']}"
    print("PASSED: pendulum coefficients recovered")

def test_fit_unknownrankscorrectly():
    ###fit unknown on pendulum data should rankpendulumform first
    pend= SimplePendulum(m=1.0, l=1.0, g=9.81)
    t,q,p=pend.generate_trajectory(q0=1.5, p0=0.0, t_span=(0,10), n_points=2000)
    H_true=pend.hamiltonian(q,p)
    fitter= ParametricHamiltonianFit()
    ranked=fitter.fit_unknown(q,p,H_true)

    print("fit_unknown ranking on pendulum (best first):")
    for r in ranked:
        if r['success']:
            print(f"  {r['name']:15s} AIC= {r['aic']:.4f} SSE = {r['sse']:.4e}")
        else:
            print(f" {r['name']:15s}  failed to fit")

    # winner should be pendulum form with SHO ranking below it
    winner = ranked[0]
    assert winner['name'] == 'pendulum_form', f"expected pendulum_form to win, so got {winner['name']}"
    assert winner['success'], "winner failed to fit"
    print("PASSED: fit_unknown identified the correct form")

def test_wrongformfailed():

    #fitting H to pendulum data should produce structurally wrong residuals

    pend  = SimplePendulum(m=1.0, l=1.0, g=9.81)
    t,q,p = pend.generate_trajectory(q0=1.5, p0=0.0, t_span=(0,10), n_points=2000)
    H_true=pend.hamiltonian(q,p)
    fitter_correct= ParametricHamiltonianFit()
    fitter_correct.fit_pendulum(q, p, H_true)
    sse_correct= float(np.sum(fitter_correct.residuals**2))

    fitter_wrong= ParametricHamiltonianFit()
    fitter_wrong.fit_sho(q,p,H_true)
    sse_wrong= float(np.sum(fitter_wrong.residuals**2))

    print(f"correct form SSE: {sse_correct:.4e}")
    print(f"wrong form SSE:  {sse_wrong:.4e}")


    if sse_correct > 0:
        sse_ratio = sse_wrong / sse_correct
        print(f"ratio (wrong/correct): {sse_ratio:.4e}")
    else:
        sse_ratio = float('inf')
        print(f"ratio (wrong/correct): inf (correct fit is at machine precision)")

    assert sse_wrong > 10.0, f"wrong form SSE is too small: {sse_wrong}"
    assert sse_ratio > 100.0, f"SSE ratio too small: {sse_ratio}"
    print("PASSED: wrong form fails as I expected")

if __name__ == "__main__":
    test_sho_recovery()
    print()
    test_pendulum_recovery()
    print()
    test_fit_unknownrankscorrectly()
    print()
    test_wrongformfailed()
    print()
    print("All parametric tests passed.")