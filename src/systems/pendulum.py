import numpy as np
from scipy.integrate import solve_ivp

class SimplePendulum:
    """Simple pendulum in Hamiltonian form
        H = p^2/2*m*1^2) + m*g*l*(1-cos(q))
        q = angle from vertical where q = 0 is hanging straight down
        p = angular momentum (p = m*1/2*dq/dt
        where
        Hamilton's eqn: dq/dt = dH/dp = p/(m*1^2)
                        dp/dt = -dH/dq = -m*g*l*sin(q)"""

    def __init__(self, m= 1.0, l= 1.0, g = 9.81):
        self.m = m
        self.l = l
        self.g = g

    def hamiltonian(self, q, p):
        return p**2/(2*self.m*self.l**2)+ self.m*self.g*self.l*(1-np.cos(q))

    def derivatives(self, t, state):
        """pendulum Hamiltonian"""

        q,p = state
        dqdt = p/(self.m*self.l ** 2)
        dpdt = -self.m * self.g * self.l *np.sin(q)
        return [dqdt, dpdt]

    def small_angle_period(self):
        """ we have small angles at t = 2*pi*sqrt(l/g)
            this is a very handwavy soln"""

        return 2*np.pi*np.sqrt(self.l/self.g)

    def generate_trajectory(self, q0, p0, t_span, n_points =1000):
        """we use dop853 as well here"""

        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(self.derivatives, t_span, [q0, p0], t_eval=t_eval, method='DOP853', rtol= 1e-12, atol=1e-12)

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        return sol.t, sol.y[0], sol.y[1]