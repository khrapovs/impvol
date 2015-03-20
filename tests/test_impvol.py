#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for Black-Scholes Implied Volatility.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import pandas as pd

from impvol import (imp_vol, find_largest_shape, lfmoneyness,
                    blackscholes_norm, impvol_table,
                    impvol_bisection, strike_from_moneyness)


class ImpVolTestCase(ut.TestCase):
    """Test Implied Volatility calculation."""

    def test_shape_finder(self):
        """Test largest shape finding function."""
        x, y = 0, 0
        self.assertEqual(find_largest_shape([x, y]), (1,))
        x, y = 0, [0, 0]
        self.assertEqual(find_largest_shape([x, y]), (2,))
        x = [0, 0]
        self.assertEqual(find_largest_shape([x, x]), (2,))
        x, y = 0, np.zeros((2, 3))
        self.assertEqual(find_largest_shape([x, y]), (2, 3))
        x, y = np.zeros((1, 3)), np.zeros((2, 3))
        self.assertEqual(find_largest_shape([x, y]), (2, 3))

    def test_moneyness(self):
        """Test conversion to moneyness."""

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

    def test_strikes(self):
        """Test conversion of moneyness to strike."""

        price, moneyness, riskfree, time = 1, 0, 0, .5
        strike = strike_from_moneyness(price, moneyness, riskfree, time)

        self.assertEqual(strike, 1)

        price, riskfree, time = 1, 0, .5
        moneyness = [1, 0]
        strike = strike_from_moneyness(price, moneyness, riskfree, time)

        np.testing.assert_array_equal(strike, np.array([np.e, 1]))

        moneyness, riskfree, time = 0, 0, .5
        price = [1, 2]
        strike = strike_from_moneyness(price, moneyness, riskfree, time)

        np.testing.assert_array_equal(strike, np.array([1, 2]))

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
        """Test values of implied volatility."""
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

    def test_bisection(self):
        """Test values of implied volatility (bisection method)."""
        premium = .024
        price = 1
        strike = 1
        riskfree = .02
        maturity = 30/365
        call = True
        moneyness = lfmoneyness(price, strike, riskfree, maturity)
        vol = impvol_bisection(moneyness, maturity, premium, call)

        self.assertAlmostEqual(float(vol), .2, 2)

        strike = [1, .95]
        premium = [.024, .057]
        moneyness = lfmoneyness(price, strike, riskfree, maturity)
        vol = impvol_bisection(moneyness, maturity, premium, call)

        np.testing.assert_array_almost_equal(vol, [.2, .2], 2)

    def test_bs_prices(self):
        """Test accuracy of back and forth BS conversion."""
        count = int(1e3)
        maturity = 30/365
        call = True
        sigma = np.random.uniform(.05, .8, count)
        moneyness = np.random.uniform(-.1, .1, count)
        premium = blackscholes_norm(moneyness, maturity, sigma, call)
        vol = impvol_bisection(moneyness, maturity, premium, call)

        np.testing.assert_array_almost_equal(vol, sigma, 5)

    def test_pandas(self):
        """Test with pandas input."""
        count = int(1e3)
        maturity = 30/365
        call = True
        sigma = np.random.uniform(.05, .8, count)
        moneyness = np.random.uniform(-.1, .1, count)
        premium = blackscholes_norm(moneyness, maturity, sigma, call)
        table = pd.DataFrame({'premium': premium, 'moneyness': moneyness})
        table['maturity'] = maturity
        table['call'] = call
        table['imp_vol'] = impvol_table(table)

        self.assertIsInstance(table, pd.DataFrame)
        np.testing.assert_array_almost_equal(table['imp_vol'].values, sigma, 5)

        dct = {'premium': premium, 'moneyness': moneyness,
               'maturity': maturity, 'call': call}
        dct['imp_vol'] = impvol_table(dct)
        self.assertIsInstance(dct, dict)
        np.testing.assert_array_almost_equal(dct['imp_vol'], sigma, 5)


if __name__ == '__main__':
    ut.main()
