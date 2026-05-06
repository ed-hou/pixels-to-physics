#DEVJournal

### Building src/models/parametric_baseline.py

I have three methods on `ParametricHamiltonianFit`:
- fit_sho: H = a*p² + b*q²
- fit_pendulum: H = a*p² + b*(1 - cos q)
- fit_unknown: which ranks candidate forms by aic and bic
All three use scipy.optimize.curve_fit which is levenger-marquardt.
The local optimizer is ok here because the froms are convex in paramaters And I seeded the initial guesses near the actual. If a future system has
a multimodal landscape I'll need multistart so won't be relevant
- The candidate hamiltonian forms, Hsho, Hpendulum, and H quartic are written as module scope rather than class methods because fit unkown need to reference them by name.
- From hte synthetic data in Sho, where m=k=1, it recovered a=.500000, b=.500000, max residual ~1e-15. 
- Pendulum where m=1=1, g=9.81 recovered a .500000 and b=9.811111 w/ max residual at ~1e-14 which is about machine precision. I will compare every neural approach in this project compared against this ceiling.


### The "wrong" experiment 
I fit the pendulum with the SHo form. The two forms agree to leading order in q (since 1-cosq is about q^2/2 for small small q), but at large angles they diverge where higher terms of cosine dominate.
Generated a trajectory at qtheta=1.5(~86degrees), which is within the nonlinear regime. Then I fit both forms to (q,p, H_true).

**result:**
- Corrected form (1-cosq): SSE -0 where we have machine precision.
- Wrong form: SSE = 56.3
- Ratio: 12 orders of magnitude

In the figure I have the left panel with the correct form and residuals flat at zero. I have the right panel with the wrong form, where residuals showing clear m shape strucutre as a function of q. 
I have zero crossings at q+-1.4, positive peaks at |q|=1 .0 and saw a neg dip at q=0.
We have a structure as a M shape, when we fit out the q^2, and what's left dominated by q^4, modulated by join distr. of (q,p)  in the trajectory data.


###gonna do neural ode

