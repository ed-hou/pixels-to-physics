import numpy as np

from scipy.integrate import solve_ivp


class SimpleHarmonicOscillator:

    """
    This is the SHO in hamiltonian form, where
    H = p^2/(2m) + (1/2)k*q^2

    and Hamiltonian's eqn's are:
    dq/dt = dH/dp = p/m
    dp/dt  = -dH/dq = -kq
    """

    def __init__(self, m = 1.0, k=1.0):
        self.m = m
        self.k = k

    def hamiltonian(self, q, p):
        """compute total energy H(q,p). constant along trajectories"""
        return p**2/(2*self.m) + .5 * self.k * q ** 2

    def derivatives(self, t, state):
        """ RHS of Hamiltonians eqns where state = [q,p]
        Returns [dq/dt, dp/dt]"""

        q,p = state
        dqdt = p/ self.m #H's 1st eqn
        dpdt = -self.k *q #H's 2nd eqn
        return [dqdt, dpdt]

    def analytical_soln(self, q0, p0, t):
        """
        soln for comparison.
        q(t) = q0 *(cos(wt)) + p0/mw)*sin(wt)
        p(t) = -q0*mw*sin*(wt) + p0*cos(wt)
        """
        w = np.sqrt(self.k/ self.m) #angular frequency
        q = q0 * np.cos(w * t) + (p0/(self.m * w)) * np.sin(w * t)
        p = -q0 * self.m * w * np.sin(w *t) + p0 * np.cos(w*t)
        return q, p

    def period(self):
        """analytical period T = 2*pi*sqrt(m/k)."""
        return 2*np.pi*np.sqrt(self.m / self.k)
    def generate_trajectory(self, q0, p0, t_span, n_points = 1000):
        """integrate Hamiltonian eqns from q0, p0
        Uses DOP853 with strict tolerance as drift would tamper with the HNN comparison later"""

        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        sol = solve_ivp(self.derivatives, t_span, [q0, p0], t_eval=t_eval, method = 'DOP853', rtol=1e-12, atol=1e-12)
        if not sol.success:
                raise RuntimeError(f"Integration failed: {sol.message}")

        return sol.t, sol.y[0], sol.y[1] #t, q, p