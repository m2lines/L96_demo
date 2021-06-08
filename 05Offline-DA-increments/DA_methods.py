""" Data assimilation methods
Adapted form PyDA project: https://github.com/Shady-Ahmed/PyDA
Reference: https://www.mdpi.com/2311-5521/5/4/225
"""
import numpy as np
from numba import jit

@jit
def Lin3dvar(ub,w,H,R,B,opt):
    
    # The solution of the 3DVAR problem in the linear case requires 
    # the solution of a linear system of equations.
    # Here we utilize the built-in numpy function to do this.
    # Other schemes can be used, instead.
    
    if opt == 1: #model-space approach
        Bi = np.linalg.inv(B)
        Ri = np.linalg.inv(R)
        A = Bi + (H.T)@Ri@H
        b = Bi@ub + (H.T)@Ri@w
        ua = np.linalg.solve(A,b) #solve a linear system 
    
    elif opt == 2: #model-space incremental approach
        
        Bi = np.linalg.inv(B)
        Ri = np.linalg.inv(R)
        A = Bi + (H.T)@Ri@H
        b = (H.T)@Ri@(w-H@ub)
        ua = ub + np.linalg.solve(A,b) #solve a linear system 
        
        
    elif opt == 3: #observation-space incremental approach
    
        A = R + H@B@(H.T)
        b = (w-H@ub)
        ua = ub + B@(H.T)@np.linalg.solve(A,b) #solve a linear system
        
    return ua

@jit
def ens_inflate(posterior,prior,opt,factor):
    
    inflated=np.zeros(posterior.shape)
    n,N=prior.shape
    if opt == "multiplicative": 
        mean_post=(posterior.sum(axis=-1)/N).repeat(N).reshape(n,N)
        inflated=posterior+factor*(posterior-mean_post)
    
    elif opt == "relaxation":
        mean_prior=(prior.sum(axis=-1)/N).repeat(N).reshape(n,N)
        mean_post=(posterior.sum(axis=-1)/N).repeat(N).reshape(n,N)
        inflated=mean_post+(1-factor)*(posterior-mean_post)+factor*(prior-mean_prior)      
        
    return inflated


@jit
def EnKF(prior,obs,H,R,B):
    
    # The analysis step for the (stochastic) ensemble Kalman filter 
    # with virtual observations

    n,N = prior.shape # n is the state dimension and N is the size of ensemble
    m = obs.shape[0] # m is the size of measurement vector
    
    mR = R.shape[0]
    nB = B.shape[0]
    mH, nH = H.shape
    assert m==mR, "observation and obs_cov_matrix have incompatible size"
    assert nB==n, "state and state_cov_matrix have incompatible size"
    assert m==mH, "obseravtion and obs operator have incompatible size"
    assert n==nH, "state and obs operator have incompatible size"

    # compute Kalman gain
    D = H@B@H.T + R
    K = B @ H.T @ np.linalg.inv(D)
            
    # perturb observations
    obs_ens=obs.repeat(N).reshape(m,N)+np.sqrt(R)@np.random.standard_normal((m,N))
    # compute analysis ensemble
    posterior = prior + K @ (obs_ens-H@prior)

    return posterior
