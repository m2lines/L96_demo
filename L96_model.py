""" Lorenz-96 model
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18.
https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved 
"""

import numpy as np 

def Lorenz_96(X,t,F,J):
    """
    Calculate the time increment in the X variables for the Lorenz '96.
    Args:
        X : Values of X variables at the current time step
        F : Forcing term
        J : Number of variables
        t : Time
    Returns:
        dXdt : Array of X time tendencies
    """
  
    s = np.zeros(J);
    s[0] = (X[1]-X[J-2])*X[J-1]-X[0];
    s[1] = (X[2]-X[J-1])*X[0]-X[1];
    s[J-1] = (X[0]-X[J-3])*X[J-2]-X[J-1];
    
    for j in range(2,J-1):
        s[j] = (X[j+1]-X[j-2])*X[j-1]-X[j];
    dXdt = s.T + F;
    return dXdt
