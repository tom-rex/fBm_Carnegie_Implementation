# A file containing all of the relevant function for our OU fbm representation

# fbm_ou.py
import numpy as np
from scipy.special import gamma

# weights for the OU superposition approximation of the kernel (t-u)^{H-1/2}
def OU_weights(H, r, n):
    a = H + 0.5
    j = np.arange(-n + 1, n + 1)
    beta = r**(j - 1)
    weights = (1/gamma(0.5 - H)) * r**((1 - a)*(j - 1)) * (r**(1 - a) - 1)/(1 - a)
    return beta, weights


#Ornstein-Uhlenbeck process simulation
""""
def simulate_OU_paths(beta, T, n_steps, M, rng, stationary_start=True):
    
    #Simulate M paths of the OU process for each rate in `beta`, jointly,
    #all driven by the SAME scalar Brownian increments per path.

    #Returns array of shape (M, n_steps+1, len(beta)).
    
    beta = np.asarray(beta)
    
    dt = T / n_steps
    decay = np.exp(-beta * dt)
    scale = np.sqrt((1 - np.exp(-2 * beta * dt)) / (2 * beta))

    Z = np.zeros((M, len(beta)))
    if stationary_start:
        Gamma = 1.0 / (beta[:, None] + beta[None, :])
        L = np.linalg.cholesky(Gamma + 1e-12 * np.eye(len(beta)))
        Z = rng.normal(size=(M, len(beta))) @ L.T

    Z_path = np.zeros((M, n_steps + 1, len(beta)))
    Z_path[:, 0, :] = Z

    for i in range(n_steps):
        noise = rng.normal(size=M)
        Z = decay[None, :] * Z + scale[None, :] * noise[:, None]
        Z_path[:, i + 1, :] = Z

    return Z_path
    """

#fractional Brownian motion simulation using OU superposition
def simulate_fBm_paths(H, r, n, T, n_steps, M, rng):
    beta, w = OU_weights(H, r, n)
    Z_path = simulate_OU_paths(beta, T, n_steps, M, rng, stationary_start=True)
    B_raw = Z_path @ w
    return B_raw - B_raw[:, [0]]

#covariance function for true fractional Brownian motion
def cov_true_fBm(s, t, H):
    return 0.5 * (s**(2*H) + t**(2*H) - abs(s - t)**(2*H))

#simulating standard Brownian motion paths
def simulate_BM_paths(T, n_steps, M, rng):
    dt = T / n_steps
    dW = rng.normal(scale=np.sqrt(dt), size=(M, n_steps))
    W = np.cumsum(dW, axis=1)
    W = np.hstack((np.zeros((M, 1)), W))
    return W

import numpy as np

def simulate_OU_paths(beta, T, n_steps, M, rng, stationary_start=True, return_brownian=False):
    """
    Simulate M paths of the OU process for each rate in `beta`, jointly,
    all driven by the SAME scalar Brownian increments per path.

    Parameters
    ----------
    beta : array_like
        Mean-reversion rates (positive).
    T : float
        Total time horizon.
    n_steps : int
        Number of time steps (discretisation).
    M : int
        Number of independent paths.
    rng : np.random.Generator
        Random number generator.
    stationary_start : bool, default=True
        If True, initialise from the stationary distribution.
    return_brownian : bool, default=False
        If True, return both OU paths and the driving Brownian motion.

    Returns
    -------
    Z_path : ndarray, shape (M, n_steps+1, len(beta))
        Simulated OU paths at each time step.
    W : ndarray, shape (M, n_steps+1), optional
        Driving Brownian motion (cumulative sum of sqrt(dt)*noise),
        starting at 0. Only returned if `return_brownian` is True.
    """
    beta = np.asarray(beta)
    dt = T / n_steps
    decay = np.exp(-beta * dt)
    scale = np.sqrt((1 - np.exp(-2 * beta * dt)) / (2 * beta))

    # Initialise Z at t=0
    Z = np.zeros((M, len(beta)))
    if stationary_start:
        Gamma = 1.0 / (beta[:, None] + beta[None, :])
        L = np.linalg.cholesky(Gamma + 1e-12 * np.eye(len(beta)))
        Z = rng.normal(size=(M, len(beta))) @ L.T

    # Store OU paths
    Z_path = np.zeros((M, n_steps + 1, len(beta)))
    Z_path[:, 0, :] = Z

    # Pre‑allocate Brownian motion (if requested)
    if return_brownian:
        W = np.zeros((M, n_steps + 1))
        sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        # Scalar noise per path, shared across all betas
        noise = rng.normal(size=M)
        Z = decay[None, :] * Z + scale[None, :] * noise[:, None]
        Z_path[:, i + 1, :] = Z

        # Accumulate Brownian motion
        if return_brownian:
            W[:, i + 1] = W[:, i] + sqrt_dt * noise

    if return_brownian:
        return Z_path, W
    else:
        return Z_path

