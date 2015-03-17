#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Black-Scholes Implied Volatility via bisection method.

"""

import numpy as np

from impvol import blackscholes_norm


def impvol_bisection(moneyness, maturity, premium, call, tol=1e-5, fcount=1e3):
    """Function to find BS Implied Vol using Bisection Method.

    Parameters
    ----------
    moneyness : float
        Log-forward moneyness
    maturity : float
        Fraction of the year
    premium : float
        Option premium normalized by current asset price
    call : bool
        Call/put flag. True for call, False for put

    Returns
    -------
    float
        Implied volatilities.
        Shape of the array is according to broadcasting rules.

    """

    sig, sig_u, sig_d = .2, 1, 1e-3
    count = 0
    err = blackscholes_norm(moneyness, maturity, sig, call) - premium

    # repeat until error is sufficiently small
    # or counter hits fcount
    while abs(err) > tol and count < fcount:
        if err < 0:
            sig_d = sig
            sig = (sig_u + sig)/2
        else:
            sig_u = sig
            sig = (sig_d + sig)/2

        err = blackscholes_norm(moneyness, maturity, sig, call) - premium
        count += 1

    # return NA if counter hit 1000
    if count == fcount:
        return -1
    else:
        return sig

# Use standard Numpy vectorization function
# The vector size is determined by the first input
vec_impvol_st = np.vectorize(impvol_bisection)
