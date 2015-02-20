#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for Black-Scholes Implied Volatility.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np

from ..impvol import imp_vol, find_largest_shape, lfmoneyness


class ImpVolTestCase(ut.TestCase):
    """Test Implied Volatility calculation."""

    def test_shape_finder(self):
        """Test largest shape finding function."""
        x, y = 0, 0
        self.assertEqual(find_largest_shape([x, y]), ())
        x, y = 0, [0, 0]
        self.assertEqual(find_largest_shape([x, y]), (2,))
        x = [0, 0]
        self.assertEqual(find_largest_shape([x, x]), (2,))
        x, y = 0, np.zeros((2, 3))
        self.assertEqual(find_largest_shape([x, y]), (2, 3))
        x, y = np.zeros((1, 3)), np.zeros((2, 3))
        self.assertEqual(find_largest_shape([x, y]), (2, 3))

    def test_moneyness(self):
        """Test conversion t moneyness."""

        price, strike, riskfree, time = 1, 1, 0, .5
        moneyness = lfmoneyness(price, strike, riskfree, time)
        self.assertEqual(moneyness, 0)

        price, riskfree, time = 1, 0, .5
        strike = [1, np.e]
        moneyness = lfmoneyness(price, strike, riskfree, time)
        np.testing.assert_array_equal(moneyness, np.array([0, 1]))

        strike, riskfree, time = 1, 0, .5
        price = [1, np.e]
        moneyness = lfmoneyness(price, strike, riskfree, time)
        np.testing.assert_array_equal(moneyness, np.array([0, -1]))

    def test_vol_types(self):
        """Test correctness of types."""

        moneyness, maturity, premium, call = 0, .3, .05, True
        vol = imp_vol(moneyness, maturity, premium, call)
        self.assertIsInstance(vol, float)

        moneyness, maturity, premium, call = [-.1, .1], .3, .05, True
        vol = imp_vol(moneyness, maturity, premium, call)
        self.assertIsInstance(vol, np.ndarray)
        self.assertEqual(vol.shape, (2,))

    def test_vol_values(self):
        """Test valies of implied volatility."""
        premium = .024
        price = 1
        strike = 1
        riskfree = .02
        maturity = 30/365
        call = True
        moneyness = lfmoneyness(price, strike, riskfree, maturity)
        vol = imp_vol(moneyness, maturity, premium, call)
        self.assertAlmostEqual(vol, .2, 2)

        strike = [1, .95]
        premium = [.024, .057]
        moneyness = lfmoneyness(price, strike, riskfree, maturity)
        vol = imp_vol(moneyness, maturity, premium, call)
        np.testing.assert_array_almost_equal(vol, [.2, .2], 2)


if __name__ == '__main__':
    ut.main()
