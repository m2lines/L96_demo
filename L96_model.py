""" Lorenz-96 model
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18.
https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved 
"""

import numpy as np 

def L96_eq1_xdot(X, F):
    """
    Calculate the time rate of change for the X variables for the Lorenz '96, equation 1:
        d/dt X[k] = -X[k-2] X[k-1] + X[k-1] X[k+1] - X[k] + F

    Args:
        X : Values of X variables at the current time step
        F : Forcing term
    Returns:
        dXdt : Array of X time tendencies
    """

    K = len(X)
    Xdot = np.zeros(K)
    
    for k in range(K):
        Xdot[k] = ( X[(k+1)%K] - X[k-2] ) * X[k-1] - X[k] + F
    return Xdot

def L96_2t_xdot_ydot(X, Y, F, h, b, c):
    """
    Calculate the time rate of change for the X and Y variables for the Lorenz '96, two time-scale
    model, equations 2 and 3:
        d/dt X[k] =     -X[k-1] ( X[k-2] - X[k+1] )   - X[k] + F - h.c/b sum_j Y[j,k]
        d/dt Y[j] = -b c Y[j+1] ( Y[j+2] - Y[j-1] ) - c Y[j]     + h.c/b X[k]

    Args:
        X : Values of X variables at the current time step
        Y : Values of Y variables at the current time step
        F : Forcing term
        h : coupling coefficient
        b : ratio of amplitudes
        c : time-scale ratio
    Returns:
        dXdt, dYdt : Array of X and Y time tendencies
    """

    JK,K = len(Y),len(X)
    J = JK//K
    assert JK==J*K, "X and Y have incompatible shapes"
    Xdot = np.zeros(K)
    hcb = (h*c)/b

    Ysummed = Y.reshape((K,J)).sum(axis=-1)
    
    #Xdot = np.roll(X,1) * ( np.roll(X,-1) - np.roll(X,2) ) - X + F - hcb * Ysummed
    for k in range(K):
        Xdot[k] = ( X[(k+1)%K] - X[k-2] ) * X[k-1] - X[k] + F - hcb * Ysummed[k]
 
    #for j in range(JK):
    #        k = j//J
    #        Ydot[j] = -c * b * Y[(j+1)%JK] * ( Y[(j+2)%JK] - Y[j-1] ) - c * Y[j] + hcb * X[k]
    Ydot = -c * b * np.roll(Y,-1) * ( np.roll(Y,-2) - np.roll(Y,1) ) - c * Y + hcb * np.repeat(X,J)

    return Xdot, Ydot

# Time-stepping methods ##########################################################################################

def EulerFwd(fn, dt, X, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,F) using the Euler forward method.

    Args:
        fn : The function returning the time rate of change of model variables X
        dt : The time step
        X  : Values of X variables at the current time, t
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)

    Returns:
        X at t+dt
    """

    return X + dt * fn(X, *params)

def RK2(fn, dt, X, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,F) using the second order Runge-Kutta method.

    Args:
        fn : The function returning the time rate of change of model variables X
        dt : The time step
        X  : Values of X variables at the current time, t
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)

    Returns:
        X at t+dt
    """

    X1 = X + 0.5 * dt * fn(X, *params)
    return X + dt * fn(X1, *params)

def RK4(fn, dt, X, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,...) using the fourth order Runge-Kutta method.

    Args:
        fn     : The function returning the time rate of change of model variables X
        dt     : The time step
        X      : Values of X variables at the current time, t
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)

    Returns:
        X at t+dt
    """

    Xdot1 = fn(X, *params)
    Xdot2 = fn(X+0.5*dt*Xdot1, *params)
    Xdot3 = fn(X+0.5*dt*Xdot2, *params)
    Xdot4 = fn(X+dt*Xdot3, *params)
    return X + (dt/6.) * ( ( Xdot1 + Xdot4 ) + 2. * ( Xdot2 + Xdot3 ) )

# Model integrators #############################################################################################

def integrate_L96_1t(X0, F, dt, nt, method=RK4, t0=0):
    """
    Integrates forward-in-time the single time-scale Lorenz 1996 model, using the integration "method".
    Returns the full history with nt+1 values starting with initial conditions, X[:,0]=X0, and ending
    with the final state, X[:,nt+1] at time t0+nt*dt.
    
    Args:
        X0     : Values of X variables at the current time
        F      : Forcing term
        dt     : The time step
        nt     : Number of forwards steps
        method : The time-stepping method that returns X(n+1) given X(n)
        t0     : Initial time (defaults to 0)

    Returns:
        X[:,:], time[:] : the full history X[n,k] at times t[n]
    
    Example usage:
        X,t = integrate_L96_1t(5+5*np.random.rand(8), 18, 0.01, 500)
        plt.plot(t, X);
    """

    time, hist = t0+np.zeros((nt+1)), np.zeros((nt+1,len(X0)))
    X = X0.copy()
    hist[0,:] = X
    for n in range(nt):
        X = method( L96_eq1_xdot, dt, X, F )
        hist[n+1], time[n+1] = X, t0+dt*(n+1)
    return hist, time

def integrate_L96_2t(X0, Y0, dt, nt, F, h, b, c, t0=0, dts=0.001):
    """
    Integrates forward-in-time the two time-scale Lorenz 1996 model, using the RK4 integration method.
    Returns the full history with nt+1 values starting with initial conditions, X[:,0]=X0 and Y[:,0]=Y0,
    and ending with the final state, X[:,nt+1] and Y[:,nt+1] at time t0+nt*dt.
    
    Note the model is intergrated 
    
    Args:
        X0  : Values of X variables at the current time
        Y0  : Values of Y variables at the current time
        dt  : Separation in time between output samples
        nt  : Number of sample segments (results in nt+1 samples incl. initial state)
        F   : Forcing term
        h   : coupling coefficient
        b   : ratio of amplitudes
        c   : time-scale ratio
        t0  : Initial time (defaults to 0)
        dts : The actual time step. If dts<dt, the dt is used. Otherwise dt/dts must be a whole number. Default 0.001.

    Returns:
        X[:,:], Y[:,:], time[:] : the full history X[n,k] and Y[n,k] at times t[n]

    Example usage:
        X,Y,t = integrate_L96_2t(5+5*np.random.rand(8), np.random.rand(8*4), 0.01, 500, 18, 1, 10, 10)
        plt.plot( t, X);
    """

    time, xhist, yhist = t0+np.zeros((nt+1)), np.zeros((nt+1,len(X0))), np.zeros((nt+1,len(Y0)))
    X,Y = X0.copy(), Y0.copy()
    xhist[0,:] = X
    yhist[0,:] = Y
    if dt<dts:
        dts, ns = dt, 1
    else:
        ns = int(dt/dts+0.5)
        assert abs(ns*dts - dt)<1e-14, "dt is not an integer multiple of dts, %f, %f, %i"%(dt,dts,ns)
    for n in range(nt):
        for s in range(ns):
            # RK4 update of X,Y
            Xdot1,Ydot1 = L96_2t_xdot_ydot(X, Y, F, h, b, c)
            Xdot2,Ydot2 = L96_2t_xdot_ydot(X+0.5*dts*Xdot1, Y+0.5*dts*Ydot1, F, h, b, c)
            Xdot3,Ydot3 = L96_2t_xdot_ydot(X+0.5*dts*Xdot2, Y+0.5*dts*Ydot2, F, h, b, c)
            Xdot4,Ydot4 = L96_2t_xdot_ydot(X+dts*Xdot3, Y+dts*Ydot3, F, h, b, c)
            X = X + (dts/6.) * ( ( Xdot1 + Xdot4 ) + 2. * ( Xdot2 + Xdot3 ) )
            Y = Y + (dts/6.) * ( ( Ydot1 + Ydot4 ) + 2. * ( Ydot2 + Ydot3 ) )

        xhist[n+1], yhist[n+1], time[n+1] = X, Y, t0+dt*(n+1)
    return xhist, yhist, time
