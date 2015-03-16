#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Greeks for option pricing
=========================

"""
from __future__ import print_function, division

import scipy.stats as scs

__all__ = ['vega']


def vega(moneyness, maturity, sigma):
    """Vega of an option with unit current underlying price.

    .. math::

        Vega_{t}\left(\tau,x\right)=S_{t}\phi\left(d_{1}\right)\sqrt{\tau}

    Log-forward moneyness:

    .. math::

        x=\log\left(K/F\right)=\log\left(K/S\right)-r\tau

        d_{1}=-\frac{x}{\sigma\sqrt{\tau}}+\frac{1}{2}\sigma\sqrt{\tau}

    Parameters
    ----------
    moneyness : array_like
        Log-forward moneyness
    maturity : array_like
        Fraction of the year, i.e. = 30/365
    sigma : array_like
        Volatility

    Returns
    -------
    array_like
        Vega of the option

    """
    sq_maturity = maturity**.5
    int_vol = sigma * sq_maturity
    return sq_maturity * scs.norm.pdf(int_vol/2 - moneyness/int_vol)
