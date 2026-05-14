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


## May 6 bug in metrics module and tests

After writing test_parametric.py, I hit two bugs that share a single root cause where synthetic data with
a perfect candidate produces a sse at 0 in float and code that assumed sse>0 crashes
1) The first bug rejected when sse=0, triggered when fit_unknown evaluated AIC for the the pendulum on pendulum data. So fixed rejecting only neg sse and return -inf when 0. I note lim aic as sse goes to 0+ is -inf. 
2) zerodivision error in test_wrongformfailed where sse_wrong/ssecorrect erroered when sse_correct=0 so I assigned float(inf) when the correct fit sse is at around machine precision.
3) Synth data would trigger both bugs

###tests suites
1) test_sho.py energy conservation, period, and the match against analytical solution
2) test_pendulum.py energy conservation, small-angle period, SHO reduction limit
3) test_parametric.py SHO recovery, pend recovery, fit_unknown ranker, and wrong form failure test

###retroactive thoughts
Easy: the curve_fit on well known probllems where the pendulum coeff recovery to machine precision was observed.
Harder: handling test edge cases.
I should have wrote test_parametric.py before the wrong form experiment. I had cascading 0 related issues. 


## May 7

Running into the issue TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead., 
Can't seem to figure out so running on CPU instead. Adding torch.set_dfault_dtype(torch.float32) does not seem to work at all after importing torch. will figure out later.

## OG Neural ODE on Simple Harmonic Oscillator

neural_ode.py is a 2, 64,64,2 multiplayer perceptron(Mlp) with tanh. RELU would fight the ode solver and produce nonsmooth phase portraits.
HamiltonianFunc will use the same architecture.

Training: adam at lr=1e-3, 800 epochs, with batch of 16 traj and window lengths of 64 timesteps. Loss curve dropped from 1e-1 to 1e-5. 
I observed on the 50s rollout that the short term tracking is ok with around 5.65 percent rel drift over the period. Given enough time the trajectory would drift off the constant H surface, so 
the og ode would not renderthe concept of energy conservation.

I expect pendulum data to be worse on this system. 

##ogneural ode on pendulum

I duplicated the same train node on train node pendulum and in paralleldid the same for ogneural diago to ogneuralpendulum. 
Used the same architecture init window, batch size, same 200 epochs and loss functions
Results: final epoch loss was 1.40 -7. The loss curve dropped to about 3.4e-2 at epoch 0 and by epoch 30 was sub 1e-6  and then plateud around 1ee-7 with normal sgd noise. 
Observed no instablility. 
For the rollout, my initial cond was q0=1.2, p0=0.0 where h0=6.255. Picked release at 69deg to put the rollout in nonlinear regime where we are out of small angle approx regime. 

Looking at the energy plot, H(t) drifts monotonically in a noisy fashion. Loiuville's theorem forbids a nonzero divergence in the learned vector field
but the mlp possesses no such constraints as it just fits the training data only. It has none of that "intuition". HNN will possess that "intuition"

50s panel shows the trajectory thickening inward into a slow spiral where a constant H curve becomes dissipative under the trained dynamics. Right edge q intercept crept from 1.2 to about 1.05

vs. SHO. Shows a larger 2.6x gap. HNN should look more dramatic on pendulum than on simple harmonic oscillator.


### May 8 

I implemented the HNN, scalar H_theta(q,p) net, 2,200,200,1 with tanh and autograd through in order to dervie dq/dt= dH/dp,
dp/dt=-dH/dq. 

Training: implemented 400 epochs. The curriculum is as follows: WINDOW=2 for epochs 0-50, window=5 from 5-150 and window=10 for 150-400.
warmup over first 500 optim steps then constant

epoch 0 loss was 2.3e-4. Each window transition created loss spike from 9e-9 to 9.9e-6 and recovered within 10 epochs. On w=5 to w=10, the loss spike wnet from 3.5e-8
to 9.8 e-6 and didn't fully recover after that. W=10 phase plateued around 1e-7 to 4e-7 and final epoch 1.5e-7 which matches ogneuralODE final sho loss.

considered cosine annealing to address window=10 plateau but after diag determined it was unncessary

Diagnostic: where q0=1.0, p0=.5, h=.625, saw .1163% energy drift whereas ogneural ode had 5.6539 percent.
This means there was a 49x improvement for energy drift from the HNN in relation to the ogneuralode

H(t) plot for ogneuralode traced a stair step decline from .625 to .590 whereas hnn oscillated symetrically around .625 with no deterministic trend. 


For pendulum on hnn, I duplicated the same process from sho. On the training I run, I ran 400 epochs.
Noticed epoch 0 loss was higher than that of sho's at 2.3e-4. noticed more loss around w=2 to w=5 at epoch 50. At around epoch150, the noise level plateaued at a higher 
level than sho did. 

NOticed about a 15.6x improvement on energy drift vs ogneuralode. HNN on pendulum and sho shows decisive imporments. 


Leapfrog integrator integrator comparison

Implemented leapfrog symplectic integrator in src/models/symplectic_integrator.py. Ran comparison diagnostics for both sho and pendulum for og
neuralode vs hnndopri5 and hnn leapfrog on 50second rollouts.

I oopsied a bug where on the kickstep2 I evaluated dh/dq to (q, p_half) instead of (q_new, p_half). This typo generated a 300% drift and
I thought there was an architectural issue. fixed the typo and the error was quickly fixed.

Results

SHO on ogneuralode 5.65 percent
SHO on HNN dopri5 .12 percent
SHo on HNN leaprfrog .1012 percent

Pendulum on ogneuralode 14.49percent
on HNNdopri5 .93%
on HNN leapfrog .9133 percent
Improvement on leapfrog is marginal. Avg about 16 percent improvement and 2% on pendulum.

SHO is closer to being an exacctly seperable ode so leapfrog's symplectic assumption fits H(theta) better, relatively. 

##KAN SHO Training and Results

KAN: width=2,5,5,1, grid=5, k=3, lamb=.001, lamb_entropy=2.0, LBFGS 100 steps. supervision-44.8k/11.2k tests points pointwise with cpu
Training: first pass loss settled at around 4.63e-3 at around step 36. Post prune, train loss dropped to 5.95e-4

Attempt 1
- (0,0,0) x², r²=1.0000 
- (0,1,0) x², r²=1.0000  
- (1,0,0) x², r²=0.9994  #wrong 
- (2,0,0) x,  r²=0.9769  

Outputted a quartic looking formula with nonzero origin offset. The mean rel error was 9.7%. H(0,0) gave -.286 when it was 0.

Observed that edge 1,0,0 picked x^2 with r^2 =.999 when in fact the truth was linear. Turned the representation wrong and quartic. So I overrode
and did model.fix_symbolic(1,0,0,'x')  with 50 lbfgs refit steps

Attempt2

Resulted in a train loss of 8.40e-5, reg=0. Formula spitted out with H= .5000*q^2+.4998p^2+ 2e-6 with mean rel erro at .0047%


