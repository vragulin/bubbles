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
    market = Market(traders=traders, earnings_ann=earnings_curr, hist_rets=hist_rets, sigma=sigma)

    assert market.traders == traders
    assert market.earnings_ann == earnings_curr
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
    market = Market(traders=traders, earnings_ann=earnings_curr, hist_rets=hist_rets, sigma=sigma)

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
    market = Market(traders=traders, earnings_ann=earnings_curr, hist_rets=hist_rets, sigma=sigma)

    price = market.equilibrium_price()
    exp_price = 0.857632972
    assert pt.approx(price, rel=1e-6) == exp_price


# @pt.mark.skip("Not implemented")
@pt.mark.parametrize("prev_weight", [0.8, None])
# @pt.mark.parametrize("prev_weight", [0.8])
def test_imbalances(prev_weight):
    traders = [
        Trader(style=VALUE, cash=0.2, shares=1.0, gamma=2.0),
        Trader(style=EXTRAPOLATOR, cash=0.45, shares=2.0, gamma=3, params=EXTRAP_PARAMS, prev_weight=prev_weight)
    ]
    earnings_curr = 0.06
    hist_rets = [0.04, 0.04, 0.04, 0.04, 0.04]
    sigma = 0.16
    market = Market(traders=traders, earnings_ann=earnings_curr, hist_rets=hist_rets, sigma=sigma)
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


def test_pay_dividends():
    # Initialize traders
    trader1 = Trader(style=0, cash=100, shares=10)
    trader2 = Trader(style=1, cash=200, shares=20, params=EXTRAP_PARAMS)
    traders = [trader1, trader2]

    # Initialize market
    market = Market(traders=traders, earnings_ann=120, earnings_month=10, payout_ratio=0.5, hist_rets=[0.1, 0.2],
                    sigma=0.3)

    # Calculate expected cash after dividends
    total_dividends = market.earnings_month * market.payout_ratio
    total_shares = trader1.shares + trader2.shares
    expected_cash_trader1 = trader1.cash + (trader1.shares / total_shares) * total_dividends
    expected_cash_trader2 = trader2.cash + (trader2.shares / total_shares) * total_dividends

    # Call pay_dividends
    market.pay_dividends()

    # Assert the cash has been updated correctly
    assert trader1.cash == expected_cash_trader1
    assert trader2.cash == expected_cash_trader2
