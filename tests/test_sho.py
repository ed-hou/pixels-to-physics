from termios import TIOCM_CAR

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.systems.sho import SimpleHarmonicOscillator

def test_energy_conservation():
    """H should be constant along the trajectory to around 1e-10."""
    sho = SimpleHarmonicOscillator(m=1.0, k = 1.0)
    t, q, p = sho.generate_trajectory(q0 = 1.0, p0 = 0.0, t_span=(0, 20), n_points=20000)

    H = sho.hamiltonian(q, p)
    H_drift = np.max(np.abs(H-H[0]) / np.abs(H[0]))

    print(f"Energy drift (relative): {H_drift: .2e}")
    assert H_drift <1e-10, f"Energy drift too large: {H_drift}"
    print("PASSED: energy conversation")

def test_period():
    """Trajectory should complete one cycle in T = 2*pi*sprt(m/k). We find the first time q returns to its initial value with the same sign of p"""
    sho = SimpleHarmonicOscillator(m=1.0, k = 1.0)
    T_Analytical = sho.period()

    #initiate at q0 = 1, p0=0
    t,q, p = sho.generate_trajectory(q0 = 1.0, p0 = 0.0, t_span=(0, 20), n_points=10000)
    #make sure there's 0 crossings of q going positive
    crossings = []

    for i in range(1, len(q)):
        if q[i - 1] < 0 and q[i] >= 0:
            frac = -q[i - 1] / (q[i] - q[i-1])
            t_cross = t[i-1] + frac * (t[i] - t[i - 1])
            crossings.append(t_cross)
    if len(crossings) == 0:
        raise RuntimeError("No crossings found")

    T_numerical = crossings[1] - crossings[0]
    error = abs(T_numerical - T_Analytical) / T_Analytical

    print(f"Analytical period: {T_Analytical:.6f}")
    print(f"Numerical period: {T_numerical:.6f}")
    print(f"Realtive error: {error:.2e}")
    assert error < 1e-6, f"Period error too large: {error}"
    #regression test to test for the unaccounted error
    sho2 = SimpleHarmonicOscillator(m=2.0, k = 1.0)
    T_actual = 2*np.pi*np.sqrt(2.0)
    T_expected = sho2.period()

    formula_error = abs(T_actual - T_expected) / T_expected
    print(f"Period with m=2, k=1, I expected {T_expected:.6f}, but got {T_actual:.6f}")
    assert formula_error < 1e-12, f"period() formula bug: {T_actual} vs  {T_expected}"
    print("PASSED where nonunit paramater works ok")
    print("PASSED, period matches analytical")

def test_against_analytical():
    """Compare numerical solution against exact closed-form soln."""
    sho = SimpleHarmonicOscillator(m=1.0, k = 1.0)
    q0, p0 = 1.0, 0.5

    t, q_num, p_num = sho.generate_trajectory(q0 = q0, p0 = p0, t_span=(0, 10), n_points=1000)
    q_exact, p_exact = sho.analytical_soln(q0, p0, t)
    q_error = np.max(np.abs(q_num - q_exact))
    p_error = np.max(np.abs(p_num - p_exact))

    print (f"Max q error : {q_error: .2e}")
    print (f"Max p error : {p_error: .2e}")
    assert q_error < 1e-9, f"q error too large: {q_error}"
    assert p_error < 1e-9, f"p error too large: {p_error}"
    print ("PASSED: MATCHES ANALYTICAL SOLUTION")

if __name__ == "__main__":
     test_energy_conservation()
     print()
     test_period()
     print()
     test_against_analytical()
     print()
     print("ALL SHO tests passed")

