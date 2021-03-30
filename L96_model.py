""" Lorenz-96 model
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18.
https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved 
"""

import numpy as np 

def L96_eq1_xdot(X,t,F):
    """
    Calculate the time rate of change for the X variables for the Lorenz '96, equation 1:
        d/dt X[k] = -X[k-2] X[k-1] + X[k-1] X[k+1] - X[k] + F

    Args:
        X : Values of X variables at the current time step
        t : Time
        F : Forcing term
    Returns:
        dXdt : Array of X time tendencies
    """

    K = len(X)
    Xdot = np.zeros(K)
    
    for k in range(K):
        Xdot[k] = ( X[(k+1)%K] - X[k-2] ) * X[k-1] - X[k] + F
    return Xdot

# Time-stepping methods ##########################################################################################

def EulerFwd(fn, dt, X, t, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,F) using the Euler forward method.
    Args:
        fn : The function returning the time rate of change of model variables X
        dt : The time step
        X  : Values of X variables at the current time, t
        t  : Time at beginning of time step
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)
    Returns:
        X at t+dt
    """
    return X + dt * fn(X, t, *params)

def RK2(fn, dt, X, t, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,F) using the second order Runge-Kutta method.
    Args:
        fn : The function returning the time rate of change of model variables X
        dt : The time step
        X  : Values of X variables at the current time, t
        t  : Time at beginning of time step
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)
    Returns:
        X at t+dt
    """
    X1 = X + 0.5 * dt * fn(X, t, *params)
    return X + dt * fn(X1, t+0.5*dt, *params)

def RK4(fn, dt, X, t, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,...) using the fourth order Runge-Kutta method.
    Args:
        fn     : The function returning the time rate of change of model variables X
        dt     : The time step
        X      : Values of X variables at the current time, t
        t      : Time at beginning of time step
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)
    Returns:
        X at t+dt
    """
    Xdot1 = fn(X, t, *params)
    Xdot2 = fn(X+0.5*dt*Xdot1, t+0.5*dt, *params)
    Xdot3 = fn(X+0.5*dt*Xdot2, t+0.5*dt, *params)
    Xdot4 = fn(X+dt*Xdot3, t+dt, *params)
    return X + (dt/6.) * ( ( Xdot1 + Xdot4 ) + 2. * ( Xdot2 + Xdot3 ) )

# Model integrators #############################################################################################

def integrator_1d(fn, method, dt, X0, nt, *params):
    """
    Integrates forward-in-time the model "fn" using the integration "method". Returns the full history with
    nt+1 values including initial conditions for n=0. The model "fn" is required to have one vector of state
    variables, X, and take the form fn(X, t, *params) where t is current model time.
    
    Args:
        fn     : The function returning the time rate of change of model variables X
        method : The time-stepping method that returns X(n+1) givein X(n)
        dt     : The time step
        X0     : Values of X variables at the current time
        nt     : Number of forwards steps
        params : All other arguments that should be passed to fn
    Returns:
        X[:,:], time[:] : the full history X[n,k] at times t[n]
    """    
    time, hist = np.zeros((nt+1)), np.zeros((nt+1,len(X0)))
    X = X0.copy()
    hist[0,:] = X
    for n in range(nt):
        X = method( fn, dt, X, n*dt, *params )
        hist[n+1], time[n+1] = X, dt*(n+1)
    return hist, time
