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


def Phi(val):
    """Standard Normal distribution with math library only.

    Parameters
    ----------
    val : float
        Argument of the CDF

    Returns
    -------
    float
        Normal CDF

    """
    return (1 + erf(val / sqrt(2))) / 2


@np.vectorize
def BSst(moneyness, maturity, vol, call):
    """Standardized Black-Scholes Function.

    .. math::
        \tilde{BS}\left(X,\sigma,T\right)
            &=	\Phi\left(d_{1}\right)-e^{X}\Phi\left(d_{2}\right),
        d_{1} &= -\frac{X}{\sigma\sqrt{T}}+\frac{1}{2}\sigma\sqrt{T},
        d_{2} &= d_{1}-\sigma\sqrt{T}.

    .. math::
        X = \log(K/S) - r*T

    Parameters
    ----------
    moneyness : float array
        Log-forward moneyness
    maturity : float array
        Fraction of the year, i.e. = 30/365
    vol : float array
        Annualized volatility (sqrt of variance), i.e. = .15
    call : bool array
        Call/put flag. True for call, False for put

    Returns
    -------
    float array
        Option premium

    """
    sqrt_matur = sqrt(maturity)
    accum_vol = vol*sqrt_matur
    d1 = - moneyness / accum_vol + accum_vol/2
    d2 = d1 - accum_vol
    if call:
        return Phi(d1) - exp(moneyness)*Phi(d2)
    else:
        return exp(moneyness)*Phi(-d2) - Phi(-d1)


@np.vectorize
def BS(S, K, T, r, sig, call):
    """Black-Scholes Function."""
    X = lfmoneyness(S, K, r, T)
    return S * BSst(X, T, sig, call)


def lfmoneyness(price, strike, riskfree, maturity):
    """Compute log-forward moneyness.

    .. math::
        X = \log(K/S) - r*T

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
