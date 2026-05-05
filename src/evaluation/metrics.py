import numpy as np

def mse_trajectory(pred, true):
    """
    returns scalar mse avg over all elements
    **averages over time and state dimensions
    """

    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs true {true.shape}")
    return float(np.mean((pred-true)**2))

def relativel2_error(pred, true):
    """scale invariant mse"""
    pred=np.asarray(pred)
    true=np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs {true.shape}")
    num = np.linalg.norm(pred - true)
    den = np.linalg.norm(true)
    if den == 0:
        raise ValueError("true trajcetory has zero norm, can't compute relative error")
    return float(num/den)

def energy_drift(H_values):
    """
    where we have maximum rel dev of H along a trajectory: max |H(T)-H(0)|/H(0)|

    for a pure hamiltonian system energy is conserved along the true trajectory so we are looking at variation
    this returns scalar relative drift
    """

    H_values = np.asarray(H_values)
    if H_values.ndim != 1:
        raise ValueError(f"expected 1D array of H values, got shape {H_values.shape}")
    H0 = H_values[0]
    if H0 == 0:
        return float(np.max(np.abs(H_values)))
    return float(np.max(np.abs(H_values - H0))/np.abs(H0))

def aic(sse, n_data, n_params):
    """
    we use Akaike Information criterion to compare models where aic=n*log(sse/n)+2*k where lower is better.
    sse is the sum of squared erros and is scalar
    n_data is the number of data points used in the fit
    n_params is the number of free params in teh model
    """

    if sse <=0:
        raise ValueError(f"sse must be positive, got {sse}")
    if n_data <=0:
        raise ValueError(f"n_data must be positive, got {n_data}")
    return float(n_data*np.log(sse/n_data)+2*n_params)
def bic(sse, n_data, n_params):
    """
    here we use the bayesian information crieterion to select models where bic=n*log(sse/n)+k*log(n) which penalizes >paramaters
    heavily.
    """

    if sse <=0:
        raise ValueError(f"sse must be positive, got {sse}")
    if n_data <=0:
        raise ValueError(f"n_data must be positive, got {n_data}")
    return float(n_data*np.log(sse/n_data)+n_params*np.log(n_data))


if __name__ == '__main__':
    #mse_trajectory where identical arrays should give 0
    a=np.array([[1.0,2.0], [3.0, 4.0]])
    assert mse_trajectory(a,a) == 0.0
    #put every element off by 1
    b=a+1.0
    assert mse_trajectory(a,b) == 1.0
    print(f"mse_trajectory: {mse_trajectory(a,b)} (expected 1.0")

    #relativel2_error
    assert relativel2_error(a,a) == 0.0
    #a 10 percent pertrubation should also give 10 relative error
    c=a*1.1
    err = relativel2_error(c,a)
    print(f"relativel2_error (10% scaling): {err:.4f} (expected ~.1)")
    assert abs(err - 0.1 < 1e-10)

    #energy_drift where constant H should be zero
    H_constant=np.ones(100)*5.0
    assert energy_drift(H_constant) == 0.0
    #pretend h drifting by 1 percent of initial value
    H_drift = np.array([1.0, 1.001, 1.005, 1.01, 1.008])
    drift = energy_drift(H_drift)
    print(f"energy_drift (max 1% deviation): {drift:.4f} (expected 0.01)")
    assert abs(drift - .01) < 1e-10

    #test aic and bic
    val_aic= aic(sse=.5, n_data=100, n_params=2)
    val_bic= bic(sse=.5, n_data=100, n_params=2)
    print(f"AIC(sse=.5, n=100, k=2): {val_aic:.4f}")
    print(f"BIC(ssse=.5, n=100, k=2): {val_bic:.4f}")
    assert val_bic >val_aic

    print("\nall metrics sanity checks passed aok")

