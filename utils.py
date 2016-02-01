import numpy as np
import scipy as sc
from scipy.special import erf

def rotate(F1,F2,c=[0,0],t=np.pi/4.):
    assert len(F1.shape) == 1
    R = np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])
    F = np.vstack([F1-c[0],F2-c[1]])
    F = np.dot(R,F)
    return F[0,:]+c[0],F[1,:]+c[1]

def phi(x,mu=0,sd=1):
    """
    Cumulative Gaussian function evaluated at x for parameters mu, sd
    """
    return 0.5 * (1 + erf((x - mu) / (sd * np.sqrt(2))))


def frange(x, y, jump):
    output = []
    while x < y:
        output.append(x)
        x += jump
    return output