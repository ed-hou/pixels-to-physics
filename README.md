# Pixels to Physics

Observing governing eqn from trajectory data by comparing four different architectures.

We first look at the parametric regression baseline, then have the og neural ode then the hamiltonian neural network and for the grand finale, I examine the komogrov arnold network

The project investigates whether prebaked physics informed biases improve performance of ML models, whether human interpretability affects accuracy, and the tradeoffs (if any)of each architecture. The KAN architecture
where it attempts symbolic discovery of Hamiltonian functions from series of data.
My main goal is to take a video of a pendulum and have the system output 
H=p^2/(2ml^2) + mgl(1-costheta)

##Done right now

1) Synth data for sho and simple pendulum with customizable noise levels and train/val/test splits
2) Parametric hamiltonian fitter via curve_fit
3) Wrong form test demonstraton
4) test suite
TODO:
1) og neural ode
2) HNN
3) KAN
4) comparison framework and then video

#I am exploring the qualities of the KAN 



##Done 
1) Synth data for SHO and simple pendulum with adjustable train/val/test splits
2) paramatric hamiltonian fitter via curve fit and wrong form failure demo
3) test suite 
4) ogneuralode on sho and pendulum wiht 5.65 and 14.49 percent drift over 50seconds respectivley