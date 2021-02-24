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
