""" Data assimilation methods
Partly adapted form PyDA project: https://github.com/Shady-Ahmed/PyDA
Reference: https://www.mdpi.com/2311-5521/5/4/225
"""
import numpy as np
from numba import njit

def observation_operator(K, l_obs, t_obs, i_t):
    """Observation operator to map between model and observation space,
    assuming linearity and model space observations.

    Args:
        K: spatial dimension of the model
        l_obs: spatial positions of observations on model grid
        t_obs: time positions of observations
        i_t: the timestep of the current DA cycle

    Returns:
        Operator matrix (K * observation_density, K)
    """
    n = l_obs.shape[-1]
    H = np.zeros((n, K))
    H[range(n), l_obs[t_obs == i_t]] = 1
    return H

def get_dist(i, j, K):
    """Compute the absolute distance between two element indices
    within a square matrix of size (K x K)

    Args:
        i: the ith row index
        j: the jth column index
        K: shape of square array

    Returns:
        Distance
    """
    return abs(i - j) if abs(i - j) <= K / 2 else K - abs(i - j)

def gaspari_cohn(distance, radius):
    """Compute the appropriate distance dependent weighting of a
    covariance matrix, after Gaspari & Cohn, 1999 (https://doi.org/10.1002/qj.49712555417)

    Args:
        distance: the distance between array elements
        radius: localization radius for DA

    Returns:
        distance dependent weight of the (i,j) index of a covariance matrix
    """
    if distance == 0:
        weight = 1.0
    else:
        if radius == 0:
            weight = 0.0
        else:
            ratio = distance / radius
            weight = 0.0
            if ratio <= 1:
                weight = (
                    -(ratio**5) / 4
                    + ratio**4 / 2
                    + 5 * ratio**3 / 8
                    - 5 * ratio**2 / 3
                    + 1
                )
            elif ratio <= 2:
                weight = (
                    ratio**5 / 12
                    - ratio**4 / 2
                    + 5 * ratio**3 / 8
                    + 5 * ratio**2 / 3
                    - 5 * ratio
                    + 4
                    - 2 / 3 / ratio
                )
    return weight

def localize_covariance(B, loc=0):
    """Localize the model climatology covariance matrix, based on
    the Gaspari-Cohn function.

    Args:
        B: Covariance matrix over a long model run 'M_truth' (K, K)
        loc: spatial localization radius for DA

    Returns:
        Covariance matrix scaled to zero outside distance 'loc' from diagonal and
        the matrix of weights which are used to scale covariance matrix
    """
    M, N = B.shape
    X, Y = np.ix_(np.arange(M), np.arange(N))
    dist = np.vectorize(get_dist)(X, Y, M)
    W = np.vectorize(gaspari_cohn)(dist, loc)
    return B * W, W

def running_average(X, N):
    """Compute running mean over a user-specified window.

    Args:
        X: Input vector of arbitrary length 'n'
        N: Size of window over which to compute mean

    Returns:
        X averaged over window N
    """
    if N % 2 == 0:
        N1, N2 = -N / 2, N / 2
    else:
        N1, N2 = -(N - 1) / 2, (N + 1) / 2
    X_sum = np.zeros(X.shape)
    for i in np.arange(N1, N2):
        X_sum = X_sum + np.roll(X, int(i), axis=0)
    return X_sum / N

def find_obs(loc, obs, t_obs, l_obs, period):
    """NOTE: This function is for plotting purposes only."""
    t_period = np.where((t_obs[:, 0] >= period[0]) & (t_obs[:, 0] < period[1]))
    obs_period = np.zeros(t_period[0].shape)
    obs_period[:] = np.nan
    for i in np.arange(len(obs_period)):
        if np.any(l_obs[t_period[0][i]] == loc):
            obs_period[i] = obs[t_period[0][i]][l_obs[t_period[0][i]] == loc]
    return obs_period

@njit
def Lin3dvar(ub, w, H, R, B, opt):
    # The solution of the 3DVAR problem in the linear case requires
    # the solution of a linear system of equations.
    # Here we utilize the built-in numpy function to do this.
    # Other schemes can be used, instead.

    if opt == 1:  # model-space approach
        Bi = np.linalg.inv(B)
        Ri = np.linalg.inv(R)
        A = Bi + (H.T) @ Ri @ H
        b = Bi @ ub + (H.T) @ Ri @ w
        ua = np.linalg.solve(A, b)  # solve a linear system

    elif opt == 2:  # model-space incremental approach
        Bi = np.linalg.inv(B)
        Ri = np.linalg.inv(R)
        A = Bi + (H.T) @ Ri @ H
        b = (H.T) @ Ri @ (w - H @ ub)
        ua = ub + np.linalg.solve(A, b)  # solve a linear system

    elif opt == 3:  # observation-space incremental approach
        A = R + H @ B @ (H.T)
        b = w - H @ ub
        ua = ub + B @ (H.T) @ np.linalg.solve(A, b)  # solve a linear system

    return ua


@njit
def ens_inflate(posterior, prior, opt, factor):
    inflated = np.zeros(posterior.shape)
    n, N = prior.shape
    if opt == "multiplicative":
        mean_post = (posterior.sum(axis=-1) / N).repeat(N).reshape(n, N)
        inflated = posterior + factor * (posterior - mean_post)

    elif opt == "relaxation":
        mean_prior = (prior.sum(axis=-1) / N).repeat(N).reshape(n, N)
        mean_post = (posterior.sum(axis=-1) / N).repeat(N).reshape(n, N)
        inflated = (
            mean_post
            + (1 - factor) * (posterior - mean_post)
            + factor * (prior - mean_prior)
        )

    return inflated


@njit
def EnKF(prior, obs, H, R, B):
    # The analysis step for the (stochastic) ensemble Kalman filter
    # with virtual observations

    n, N = prior.shape  # n is the state dimension and N is the size of ensemble
    m = obs.shape[0]  # m is the size of measurement vector

    mR = R.shape[0]
    nB = B.shape[0]
    mH, nH = H.shape
    assert m == mR, "observation and obs_cov_matrix have incompatible size"
    assert nB == n, "state and state_cov_matrix have incompatible size"
    assert m == mH, "obseravtion and obs operator have incompatible size"
    assert n == nH, "state and obs operator have incompatible size"

    # compute Kalman gain
    D = H @ B @ H.T + R
    K = B @ H.T @ np.linalg.inv(D)

    # perturb observations
    obs_ens = obs.repeat(N).reshape(m, N) + np.sqrt(R) @ np.random.standard_normal(
        (m, N)
    )
    # compute analysis ensemble
    posterior = prior + K @ (obs_ens - H @ prior)

    return posterior
