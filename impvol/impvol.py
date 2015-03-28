#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Black-Scholes Implied Volatility
================================

Introduction
------------

The original Black-Scholes formula is given by

.. math::
    BS\left(S,K,\sigma,r,T\right)
        &=S\Phi\left(d_{1}\right)-e^{-rT}K\Phi\left(d_{2}\right),\\
    d_{1}&=\frac{\log\left(S/K\right)+rT}{\sigma\sqrt{T}}
        +\frac{1}{2}\sigma\sqrt{T},\\
    d_{2}&=d_{1}-\sigma\sqrt{T}.

After normalization by the current asset price :math:`S` it can be written as

.. math::
    \tilde{BS}\left(X,\sigma,T\right)
        &=\Phi\left(d_{1}\right)-e^{X}\Phi\left(d_{2}\right),\\
    d_{1}&=-\frac{X}{\sigma\sqrt{T}}+\frac{1}{2}\sigma\sqrt{T},\\
    d_{2}&=d_{1}-\sigma\sqrt{T},

where :math:`X=\log\left(K/F\right)` is log-forward moneyness,
and forward price is given by :math:`F=Se^{rT}`.

Examples
--------
.. doctest::

    >>> from impvol import imp_vol, lfmoneyness
    >>> strike = [1, .95]
    >>> premium = [.024, .057]
    >>> price = 1
    >>> riskfree = .02
    >>> maturity = 30/365
    >>> call = True
    >>> moneyness = lfmoneyness(price, strike, riskfree, maturity)
    >>> vol = imp_vol(moneyness, maturity, premium, call)
    >>> print(vol)
    [ 0.20277309  0.20093061]
    >>> vol = impvol_bisection(moneyness, maturity, premium, call)
    >>> print(vol)
    [ 0.20270996  0.20095215]

Functions
---------

"""
from __future__ import print_function, division

import numpy as np

from scipy.optimize import root
from scipy.stats import norm

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"

__all__ = ['imp_vol', 'find_largest_shape', 'impvol_bisection', 'impvol_table',
           'lfmoneyness', 'strike_from_moneyness', 'blackscholes_norm']


def blackscholes_norm(moneyness, maturity, vol, call):
    """Standardized Black-Scholes Function.

    Parameters
    ----------
    moneyness : array_like
        Log-forward moneyness
    maturity : array_like
        Fraction of the year, i.e. = 30/365
    vol : array_like
        Annualized volatility (sqrt of variance), i.e. = .15
    call : bool array_like
        Call/put flag. True for call, False for put

    Returns
    -------
    array_like
        Option premium standardized by current asset price

    """
    accum_vol = np.atleast_1d(vol)*np.atleast_1d(maturity)**.5
    d1arg = - np.atleast_1d(moneyness) / accum_vol + accum_vol/2
    d2arg = d1arg - accum_vol
    out1 = norm.cdf(d1arg) - np.exp(moneyness)*norm.cdf(d2arg)
    out2 = np.exp(moneyness)*norm.cdf(-d2arg) - norm.cdf(-d1arg)
    premium = out1 * call + out2 * np.logical_not(call)
    return premium


def blackscholes(price, strike, maturity, riskfree, vol, call):
    """Black-Scholes function.

    Parameters
    ----------
    price : array_like
        Underlying prices
    strike : array_like
        Option strikes
    maturity : array_like
        Fraction of the year, i.e. = 30/365
    riskfree : array_like
        Annualized risk-free rate
    vol : array_like
        Annualized volatility (sqrt of variance), i.e. = .15
    call : bool array_like
        Call/put flag. True for call, False for put

    Returns
    -------
    array_like
        Option premium

    """
    moneyness = lfmoneyness(price, strike, riskfree, maturity)
    return price * blackscholes_norm(moneyness, maturity, vol, call)


def lfmoneyness(price, strike, riskfree, maturity):
    """Compute log-forward moneyness.

    Parameters
    ----------
    price : array_like
        Underlying prices
    strike : array_like
        Option strikes
    riskfree : array_like
        Annualized risk-free rate
    maturity : array_like
        Time horizons, in shares of the calendar year

    Returns
    -------
    array_like
        Log-forward moneyness

    """
    return np.log(strike) - np.log(price) - riskfree * maturity


def strike_from_moneyness(price, moneyness, riskfree, maturity):
    """Compute strike from log-forward moneyness.

    Parameters
    ----------
    price : array_like
        Underlying prices
    moneyness : array_like
        Log-forward moneyness
    riskfree : array_like
        Annualized risk-free rate
    maturity : array_like
        Time horizons, in shares of the calendar year

    Returns
    -------
    array_like
        Option strikes

    """
    return price * np.exp(moneyness) * np.exp(riskfree * maturity)


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
    out = np.array([0])
    for array in arrays:
        out = out * np.zeros_like(array)
    return out.shape


def imp_vol(moneyness, maturity, premium, call):
    """Compute implied volatility given vector of option premium.

    Parameters
    ----------
    moneyness : array_like
        Log-forward moneyness
    maturity : array_like
        Fraction of the year
    premium : array_like
        Option premium normalized by current asset price
    call : bool array_like
        Call/put flag. True for call, False for put

    Returns
    -------
    array_like
        Implied volatilities.
        Shape of the array is according to broadcasting rules.

    Notes
    -----
    This code relies on SciPy root method. Although vectorized, it is still
    very slow. Bisection method in this impvol library is substantially
    faster.

    """
    args = [moneyness, maturity, premium, call]
    start = np.ones(find_largest_shape(args)) * .2
    out = root(lambda vol: error_iv(vol, moneyness, maturity, premium, call),
               start, method='lm')
    return out.x


def error_iv(sigma, moneyness, maturity, premium, call):
    """Pricing error.

    Parameters
    ----------
    sigma : array_like
        Volatility
    moneyness : array_like
        Log-forward moneyness
    maturity : array_like
        Fraction of the year
    premium : array_like
        Option premium normalized by current asset price
    call : array_like bool
        Call/put flag. True for call, False for put

    Returns
    -------
    array_like
        Implied volatilities.
        Shape of the array is according to broadcasting rules.

    """
    # TODO : read below
    # An alternative would be to minimize the relative error:
    # blackscholes_norm(moneyness, maturity, sigma, call) / premium - 1
    # But in that case some premia may be zero. What then?
    return blackscholes_norm(moneyness, maturity, sigma, call) - premium


def impvol_bisection(moneyness, maturity, premium, call,
                     tol=1e-5, fcount=1000):
    """Function to find BS Implied Vol using Bisection Method.

    Parameters
    ----------
    moneyness : array_like
        Log-forward moneyness
    maturity : array_like
        Fraction of the year
    premium : array_like
        Option premium normalized by current asset price
    call : array_like bool
        Call/put flag. True for call, False for put

    Returns
    -------
    array_like
        Implied volatilities
        Shape of the array is according to broadcasting rules.

    """
    args = [moneyness, maturity, premium, call]
    size = find_largest_shape(args)
    sigma = np.ones(size) * .2
    sigma_u = np.ones(size)
    sigma_d = np.ones(size) * 1e-3

    count = 0
    error = [tol + 1]

    # repeat until error is sufficiently small or counter hits fcount
    while np.max(np.absolute(error)) > tol and count < fcount:

        error = error_iv(sigma, moneyness, maturity, premium, call)
        sigma_u[error >= 0] = sigma[error >= 0]
        sigma_d[error < 0] = sigma[error < 0]
        sigma += sigma_d * (error >= 0) + sigma_u * (error < 0)
        sigma /= 2
        count += 1

    return sigma


def impvol_table(data):
    """Implied volatility for structured data.

    Parameters
    ----------
    data : pandas DataFrame, record array, or dictionary of arrays
        Mandatory labels: moneyness, maturity, premium, call

    Returns
    -------
    array
        Implied volatilities

    Notes
    -----
    'premium' should be normalized by the current asset price.

    """
    try:
        data = data.to_records()
    except:
        pass
    return impvol_bisection(data['moneyness'], data['maturity'],
                            data['premium'], data['call'])


if __name__ == '__main__':
    import doctest
    doctest.testmod()
