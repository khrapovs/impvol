from math import exp, log, sqrt, erf
import numpy as np

# Standard Normal distribution with math library only
def Phi(x):
    return .5 * ( 1. + erf(x / sqrt(2)) )

@np.vectorize
def BSst(X, T, sig, call):
    """Standardized Black-Scholes Function."""
    # X = log(K/S) - r*T
    d1 = -X / (sig*sqrt(T)) + .5 * sig*sqrt(T)
    d2 = d1 - sig*sqrt(T)
    if call:
        return Phi(d1) - exp(X)*Phi(d2)
    else:
        return exp(X)*Phi(-d2) - Phi(-d1)

@np.vectorize
def BS(S, K, T, r, sig, call):
    """Black-Scholes Function."""
    X = log(K/S) - r*T
    return S * BSst(X, T, sig, call)

# Function to find BS Implied Vol using Bisection Method
def impvol(S, K, T, r, C, cp, tol = 1e-5, fcount = 1e3):
    sig, sig_u, sig_d = .2, 1., 1e-3
    count = 0
    err = BS(S, K, T, r, sig, cp) - C

    # repeat until error is sufficiently small
    # or counter hits fcount
    while abs(err) > tol and count < fcount:
        if err < 0:
            sig_d = sig
            sig = (sig_u + sig)/2
        else:
            sig_u = sig
            sig = (sig_d + sig)/2
        
        err = BS(S, K, T, r, sig, cp) - C
        count += 1
    
    # return NA if counter hit 1000
    if count == fcount:
        return -1
    else:
        return sig

def impvol_st(X, T, C, cp, tol = 1e-5, fcount = 1e3):
    """Function to find BS Implied Vol using Bisection Method."""

    sig, sig_u, sig_d = .2, 1., 1e-3
    count = 0
    err = BSst(X, T, sig, cp) - C
    
    # repeat until error is sufficiently small
    # or counter hits fcount
    while abs(err) > tol and count < fcount:
        if err < 0:
            sig_d = sig
            sig = (sig_u + sig)/2
        else:
            sig_u = sig
            sig = (sig_d + sig)/2
        
        err = BSst(X, T, sig, cp) - C
        count += 1
    
    # return NA if counter hit 1000
    if count == fcount:
        return -1
    else:
        return sig

# Use standard Numpy vectorization function
# The vector size is determined by the first input
vec_impvol_st = np.vectorize(impvol_st)

def impvol(X, T, C, call):
    """Compute implied volatility given vector of option premium C.

    The function is already vectorized since BSst is.

    Inputs:
    X - log-forward moneyness
    T - fraction of the year
    C - option premium normalized by current asset price
    """

    f = lambda sig : BSst(X, T, sig, call) - C
    return so.root(f, np.ones_like(C) * .2).x

# Test code:
# S - stock price
# K - strike
# T - maturity in years
# r - risk free rate annualized
# market - option price
#
#S, K, T, r, market, cp = 1, 1, 30./365, 0, .1, 'C'
#v = impvol(S, K, T, r, market, cp)
#print v

# Test
# X = log(K/F)
# v = vec_impvol_st(X, T, C/S, cp)