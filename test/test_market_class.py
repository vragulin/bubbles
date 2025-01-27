import pytest as pt
from src.market_class import Trader, Market, VALUE, EXTRAPOLATOR
from src.config import EXTRAP_PARAMS


def test_market_initialization():
    traders = [
        Trader(style=VALUE, cash=1.0, shares=2.0, gamma=3.0),
        Trader(style=EXTRAPOLATOR, cash=1.5, shares=1.0, gamma=2.5, params=EXTRAP_PARAMS)
    ]
    earnings_curr = 1.0
    hist_rets = [0.05, 0.04, 0.03, 0.02, 0.01]
    sigma = 0.2
    market = Market(traders=traders, earnings_curr=earnings_curr, hist_rets=hist_rets, sigma=sigma)

    assert market.traders == traders
    assert market.earnings_curr == earnings_curr
    assert market.hist_rets == hist_rets
    assert market.sigma == sigma


def test_equilibrium_price_0():
    traders = [
        Trader(style=VALUE, cash=1.0, shares=2.0, gamma=3.0),
        Trader(style=EXTRAPOLATOR, cash=1.5, shares=1.0, gamma=2.5, params=EXTRAP_PARAMS)
    ]
    earnings_curr = 1.0
    hist_rets = [0.05, 0.04, 0.03, 0.02, 0.01]
    sigma = 0.2
    market = Market(traders=traders, earnings_curr=earnings_curr, hist_rets=hist_rets, sigma=sigma)

    price = market.equilibrium_price()
    assert price > 0


def test_equilibrium_price_1():
    traders = [
        Trader(style=VALUE, cash=0.2, shares=1.0, gamma=2.0),
        Trader(style=EXTRAPOLATOR, cash=0.45, shares=2.0, gamma=3.0, params=EXTRAP_PARAMS)
    ]
    earnings_curr = 0.06
    hist_rets = [0.04, 0.04, 0.04, 0.04, 0.04]
    sigma = 0.16
    market = Market(traders=traders, earnings_curr=earnings_curr, hist_rets=hist_rets, sigma=sigma)

    price = market.equilibrium_price()
    exp_price = 0.857632972
    assert pt.approx(price, rel=1e-6) == exp_price


# @pt.mark.skip("Not implemented")
def test_imbalances():
    traders = [
        Trader(style=VALUE, cash=0.2, shares=1.0, gamma=2.0),
        Trader(style=EXTRAPOLATOR, cash=0.45, shares=2.0, gamma=3, params=EXTRAP_PARAMS)
    ]
    earnings_curr = 0.06
    hist_rets = [0.04, 0.04, 0.04, 0.04, 0.04]
    sigma = 0.16
    market = Market(traders=traders, earnings_curr=earnings_curr, hist_rets=hist_rets, sigma=sigma)
    market.price = market.equilibrium_price()

    imbalances = market.imbalances()
    assert isinstance(imbalances, dict)
    assert len(imbalances) == len(traders)
    imbalance_cash, imbalance_shares = 0, 0
    for trader, imbalance in imbalances.items():
        imbalance_cash += imbalance['cash']
        imbalance_shares += imbalance['shares']
    assert pt.approx(imbalance_cash) == 0
    assert pt.approx(imbalance_shares) == 0
