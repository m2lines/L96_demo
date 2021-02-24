""" Lorenz-96 model
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18.
https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved 
"""

import numpy as np 

def Lorenz_96(X,t,F):
    """
    Calculate the time rate of change for the X variables for the Lorenz '96.
    Args:
        X : Values of X variables at the current time step
        F : Forcing term
        t : Time
    Returns:
        dXdt : Array of X time tendencies
    """

    J = len(X)
    s = np.zeros(J)
    
    for j in range(J):
        s[j] = (X[(j+1)%J]-X[j-2])*X[j-1]-X[j]
    dXdt = s.T + F
    return dXdt

def EulerFwd(fn, dt, X, t, F):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,F) using the Euler forward method.
    Args:
        fn : The function returning the time rate of change of model variables X
        dt : The time step
        X  : Values of X variables at the current time, t
        t  : Time at beginning of time step
        F  : Forcing term
    Returns:
        X at t+dt
    """
    return X + dt * fn(X, t, F)

def RK2(fn, dt, X, t, F):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,F) using the second order Runge-Kutta method.
    Args:
        fn : The function returning the time rate of change of model variables X
        dt : The time step
        X  : Values of X variables at the current time, t
        t  : Time at beginning of time step
        F  : Forcing term
    Returns:
        X at t+dt
    """
    X1 = X + 0.5 * dt * fn(X, t, F)
    return X + dt * fn(X1, t+0.5*dt, F)

def RK4(fn, dt, X, t, F):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,F) using the fourth order Runge-Kutta method.
    Args:
        fn : The function returning the time rate of change of model variables X
        dt : The time step
        X  : Values of X variables at the current time, t
        t  : Time at beginning of time step
        F  : Forcing term
    Returns:
        X at t+dt
    """
    Xdot1 = fn(X, t, F)
    Xdot2 = fn(X+0.5*dt*Xdot1, t+0.5*dt, F)
    Xdot3 = fn(X+0.5*dt*Xdot2, t+0.5*dt, F)
    Xdot4 = fn(X+dt*Xdot3, t+dt, F)
    return X + (dt/6.) * ( ( Xdot1 + Xdot4 ) + 2. * ( Xdot2 + Xdot3 ) )

def integrator(fn, method, dt, X0, F, nt):
    """
    Integrates forward-in-time the model "fn" using the integration "method". Returns the full history.
    
    Args:
        fn     : The function returning the time rate of change of model variables X
        method : The function returning the time rate of change of model variables X
        dt     : The time step
        X0     : Values of X variables at the current time
        F      : Forcing term
        t      : Time
        nt     : Number of forwards steps
    Returns:
        X(j,time), t, the full history at times t
    """    
    time, hist = np.zeros((nt+1)), np.zeros((nt+1,len(X0)))
    X = X0.copy()
    hist[0,:] = X
    for n in range(nt):
        X = method( fn, dt, X, n*dt, F )
        hist[n+1], time[n+1] = X, dt*(n+1)
    return hist, time
