#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Examples of using implied volatility.

"""
from __future__ import print_function, division

import time

import numpy as np

from impvol import imp_vol, impvol_bisection, lfmoneyness, blackscholes_norm


if __name__ == '__main__':

    price = 1
    strike = 1
    riskfree = .02
    maturity = 30/365
    premium = .057
    call = True
    moneyness = lfmoneyness(price, strike, riskfree, maturity)
    vol = impvol_bisection(moneyness, maturity, premium, call)

    count = int(1e2)
    sigma = np.random.uniform(.05, .8, count)
    moneyness = np.random.uniform(-.1, .1, count)
    premium = blackscholes_norm(moneyness, maturity, sigma, call)

    text = 'Time elapsed: %.2f seconds'

    time_start = time.time()
    # Based on SciPy root method
    vol = imp_vol(moneyness, maturity, premium, call)
    print(np.allclose(sigma, vol))
    print(text % (time.time() - time_start))

    time_start = time.time()
    # Based on bisection method
    vol = impvol_bisection(moneyness, maturity, premium, call)
    print('Relative difference (percent) = ', ((sigma/vol-1).max()*100))
    print(np.allclose(vol, sigma, rtol=1e-3))
    print(text % (time.time() - time_start))
