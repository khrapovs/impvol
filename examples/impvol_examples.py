#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Examples of using implied volatility.

"""
from __future__ import print_function, division

import time

import numpy as np

from impvol import imp_vol, impvol_bisection, lfmoneyness


if __name__ == '__main__':

    strike = [1, .95]
    premium = [.024, .057]

    strike = 1
    premium = np.ones(int(1e3)) * .057

    price = 1
    riskfree = .02
    maturity = 30/365
    call = True
    moneyness = lfmoneyness(price, strike, riskfree, maturity)

    text = 'Time elapsed: %.2f seconds'

    time_start = time.time()
    vol = imp_vol(moneyness, maturity, premium, call)
    print(text % (time.time() - time_start))

    time_start = time.time()
    vol = impvol_bisection(moneyness, maturity, premium, call)
    print(text % (time.time() - time_start))
