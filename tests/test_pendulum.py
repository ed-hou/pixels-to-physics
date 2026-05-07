import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.systems.pendulum import SimplePendulum
from src.systems.sho import SimpleHarmonicOscillator

def test_energy_conservation():
    """H should be constant along the trajectory"""
    pend = SimplePendulum(m=1.0, l =1.0, g=9.81)
    #test at a large angle
    q0 = 1.0 #57 deg
    p0 = 0.0
    t, q, p = pend.generate_trajectory(q0, p0, t_span = (0, 20), n_points = 5000)

    H = pend.hamiltonian(q, p)
    H_drift = np.max(np.abs(H-H[0])) / np.abs(H[0])

    print(f"Energy drift (relative): {H_drift: .2e}")
    assert H_drift < 1e-10, f"Energy drift too large: {H_drift}"
    print("Passed: energy conversation")

def testsmallAnglePeriod():
    """for small initial angle, the period should match the sho approximation t= 2*pi*sqrt(l/g)"""
    pend = SimplePendulum(m=1.0, l =1.0, g=9.81)
    T_approx = pend.small_angle_period()


    q0 = 0.01
    t, q, p = pend.generate_trajectory(q0, 0.0, t_span=(0, 20), n_points = 200000)

    crossings = []
    for i in range(1, len(q)):
        if q[i-1] < 0 and q[i] >= 0:
            frac = -q[i-1] / (q[i] - q[i-1])
            t_cross = t[i-1] + frac * (t[i] - t[i-1])
            crossings.append(t_cross)

    T_numerical = crossings[1] - crossings[0]
    print(f"Number of crossings found: {len(crossings)}")
    print(f"First few crossings: {crossings[:5]}")
    error = abs(T_numerical - T_approx) / T_approx

    print(f"Small=angle period (analytical): {T_approx: .6f}")
    print(f"Small-angle period (numerical): {T_numerical: .6f}")
    print(f"Relative error: {error: .2e}")
    assert error < 1e-5, f"Period error too large: {error}"
    print("PASSED: small-angle period matches")

def testreducestoSho():
    """At small angles, pendulum trajectory should match SHO trajectory with effective spring constant k_eff = m*g*l / m*l^2) = g/l"""
    m, l, g = 1.0, 1.0, 9.81
    pend = SimplePendulum(m=m, l=l, g=g)

    sho = SimpleHarmonicOscillator(m=m*l**2, k=m*g*l)

    q0 = .01 #small angle
    p0 = 0.0
    t_span = (0, 10)

    t, q_pend, p_pend = pend.generate_trajectory(q0, p0, t_span, n_points = 2000)
    t, q_sho, p_sho = sho.generate_trajectory(q0, p0, t_span= t_span, n_points = 2000)

    q_diff = np.max(np.abs(q_pend - q_sho))
    print(f"Max diff between pendulum and SHO at small angle: {q_diff: .2e}")
    assert q_diff < 1e-4, f"Pendulum doesn't reduce to SHO: {q_diff}"
    print("PASSED: pendulum reduces to SHO at small angles")

if __name__ == "__main__":
    test_energy_conservation()
    print()
    testsmallAnglePeriod()
    print()
    testreducestoSho()
    print()
    print("ALL pendulum tests passed.")

