#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Black-Scholes Implied Volatility via bisection method.

"""

import numpy as np

from imp_vol import BSst

def impvol_st(X, T, C, cp, tol=1e-5, fcount=1e3):
    """Function to find BS Implied Vol using Bisection Method."""

    sig, sig_u, sig_d = .2, 1, 1e-3
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