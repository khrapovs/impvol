#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Black-Scholes Implied Volatility.

"""

from __future__ import print_function, division

from math import exp, sqrt, erf
from scipy.optimize import root
import numpy as np

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"


# Standard Normal distribution with math library only
def Phi(x):
    return .5 * (1 + erf(x / sqrt(2)))


@np.vectorize
def BSst(X, T, sig, call):
    """Standardized Black-Scholes Function.

    .. math::

    moneyness : float array
        log-forward moneyness
    maturity : float array
        fraction of the year
    premium : float array
        option premium normalized by current asset price
    call : bool array
        call/put flag. True for call, False for put

    """
    # X = log(K/S) - r*T
    d1 = -X / (sig*sqrt(T)) + sig*sqrt(T)/2
    d2 = d1 - sig*sqrt(T)
    if call:
        return Phi(d1) - exp(X)*Phi(d2)
    else:
        return exp(X)*Phi(-d2) - Phi(-d1)


@np.vectorize
def BS(S, K, T, r, sig, call):
    """Black-Scholes Function."""
    X = lfmoneyness(S, K, r, T)
    return S * BSst(X, T, sig, call)


def lfmoneyness(price, strike, riskfree, maturity):
    """Compute log-forward moneyness.

    Parameters
    ----------
    price : float array
        Underlying prices
    strike : float array
        Option strikes
    riskfree : float array
        Annualized risk-free rate
    maturity : float array
        Time horizons, in shares of the calendar year

    Returns
    -------
    float array
        Log-forward moneyness

    """
    moneyness = (np.log(np.atleast_1d(strike) / price)
        - np.atleast_1d(riskfree) * maturity)
    if moneyness.size == 1:
        moneyness = float(moneyness)
    return moneyness


def find_largest_shape(arrays):
    """Find largest shape among series of arrays.

    Parameters
    ----------
    arrays : list
        List of arrays

    Returns
    -------
    tuple
        Largest shape among arrays

    """
    out = np.array(0)
    for array in arrays:
        out = out * np.zeros_like(array)
    return out.shape


def impvol(moneyness, maturity, premium, call):
    """Compute implied volatility given vector of option premium C.

    The function is already vectorized since BSst is.

    Parameters
    ----------
    moneyness : float array
        log-forward moneyness
    maturity : float array
        fraction of the year
    premium : float array
        option premium normalized by current asset price
    call : bool array
        call/put flag. True for call, False for put

    Returns
    -------
    float or float array
        Implied volatilities.
        Shape of the array is according to broadcasting rules.

    """
    args = [moneyness, maturity, premium, call]
    error = lambda sig: BSst(moneyness, maturity, sig, call) - premium
    start = np.ones(find_largest_shape(args)) * .2
    vol = root(error, start, method='lm').x
    if vol.size == 1:
        vol = float(vol)
    return vol

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
