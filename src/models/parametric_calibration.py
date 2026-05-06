import numpy as np
from scipy.optimize import curve_fit

##if we know the right form, curve fitting works perfectly trivially, but if wrong guess, we observe the fit collapsing. So we guess a few candidate equations, demonstrating the need for KAN.
def _H_sho(qp,a,b):
    #H=a*p^2_b*q*2., curve_fit needs the indep vars packed into qp.
    q,p = qp
    return a*p**2+b*q**2

def _H_pendulum(qp, a, b):
    """where h = a*p^2 + b*(1-cos(q)). fitting into this pendulum model shoudl give better residuals"""
    q,p = qp
    return a*p**2+b*(1-np.cos(q))

def _H_quartic(qp, a, b, c):
    #demonstrate aic penalty
    return a*p**2+b*q**2+c*q**4

class ParametricHamiltonianFit:
    """
    first a known hamiltonian functional form to q,p,H data.
    I store the most recent fit so I can inspect popt/pcov without refitting.
    """
    def __init__(self):
        self.popt = None
        self.pcov = None
        self.form_name = None
        self.residuals = None

    def fit_sho(self, q, p, H_true):
        """
        fit h= a*p^2 + b*q^2. Expected: a=1/(2m), b=k/2
        So for m=k=1 we have a=b=.5
        """
        q,p,H = q.ravel(), p.ravel(), H_true.ravel()
        popt, pcov = curve_fit(_H_sho, (q,p), H, p0= [0.5, 0.5])
        self.popt=popt
        self.pcov=pcov
        self.form_name = 'sho'
        self.residuals = H-_H_sho((q,p), *popt)
        a,b = popt

        formula = f"H = {a:.4f}*p^2 + {b:.4f}*q^2"
        return {'a': a, 'b': b, 'formula': formula}

    def fit_pendulum(self, q, p, H_true):
        """
        fir H=a*p*2+b*(1-cos(q)). expected a=1/(2ml^2), b=mgl
        where for m=l=1 and g=9.81 and a=.5, b=9.81
        """
        q, p, H = q.ravel(), p.ravel(), H_true.ravel()

        popt, pcov = curve_fit(_H_pendulum, (q,p), H, p0=[0.5, 9.0])
        self.popt=popt
        self.pcov=pcov
        self.form_name = 'pendulum'
        self.residuals = H-_H_pendulum((q,p), *popt)
        a,b = popt
        formula = f"H = {a: .4f}*p^2 + {b:.4f}*1-cos(q))"

        return {'a': a, 'b': b, 'formula': formula}

    def fit_unknown(self, q, p, H_true, candidates=None):
        """
        Try candidate forms which is ranked by aic, where candidates will be a list of tuples.
        If none, we use the three default forms inthe form of SHO, pendulum and quartic.
        -will return a list of dicts sorted by aic ascending
        - this will let AIC pick the winner, compared to KAN
        - I want to observe a whole, comprehensive list of the sorted list to see how models fit

        """

        from src.evaluation.metrics import aic, bic

        if candidates is None:
            candidates = [('sho_form', _H_sho, [0.5, 0.5]), ('pendulum_form', _H_pendulum, [0.5, 9.0]), ('quartic_form', _H_quartic, [0.5, 0.5, 0.0]),]

        q,p,H= q.ravel(), p.ravel(), H_true.ravel()
        n_data=len(H)
        results=[]
        for name, func, p0 in candidates:
            try:
                popt, pcov=curve_fit(func, (q,p),H, p0=p0, maxfev=5000)
                resid=H-func((q,p), *popt)
                sse=float(np.sum(resid**2))
                results.append({'name': name, 'popt': popt, 'sse': sse, 'aic': aic(sse, n_data, len(p0)), 'bic': bic(sse, n_data, len(p0)), 'n_params': len(p0), 'success': True,})
            except RuntimeError as e:
                results.append({'name': name, 'popt': None, 'sse': np.inf, 'aic': np.inf, 'bic': np.inf, 'n_params': len(p0), 'success': False, 'error': str(e)})

        return sorted(results, key=lambda r: r['aic'])

if __name__ == '__main__':

        # main tests
        from src.systems.sho import SimpleHarmonicOscillator

        sho = SimpleHarmonicOscillator(m = 1.0, k=1.0)
        t, q, p = sho.generate_trajectory(q0=1.0, p0=0.5, t_span=(0,10), n_points=1000)
        H_true = sho.hamiltonian(q,p)

        fitter_bot = ParametricHamiltonianFit()
        result = fitter_bot.fit_sho(q,p, H_true)
        print(f"SHO fit: {result['formula']}")
        print(f"   expected a=.5, got a={result['a']:.6f}")
        print(f"   expected b=.5, got b={result['b']:.6f}")
        print(f"   max residual, {np.max(np.abs(fitter_bot.residuals)): .2e}")

