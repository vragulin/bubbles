import pytest
import numpy as np
from src.market_class import Trader
from src.config import EXTRAP_PARAMS, VALUE, EXTRAPOLATOR


def test_trader_initialization():
    trader = Trader(style=VALUE, cash=1.0, shares=2.0, gamma=3.0)
    assert trader.style == VALUE
    assert trader.cash == 1.0
    assert trader.shares == 2.0
    assert trader.gamma == 3.0


def test_trader_repr():
    trader = Trader(style=EXTRAPOLATOR, cash=1.0, shares=2.0, gamma=3.0, params=EXTRAP_PARAMS)
    expected_repr = "Trader(style=1, cash=1.0, stock=2.0, gamma=3.0)"
    assert repr(trader) == expected_repr


def test_merton_share():
    expected_return = 0.1
    risk_aversion = 2.0
    volatility = 0.3
    expected_share = expected_return / (risk_aversion * volatility ** 2)
    assert Trader.merton_share(expected_return, risk_aversion, volatility) == expected_share


def test_value_trader_expected_return():
    trader = Trader(style=VALUE)
    earnings = 1.0
    price = 8.0
    expected_return = earnings / price
    assert trader.expected_return(earnings=earnings, price=price) == expected_return


def test_extrapolator_trader_expected_return_tanh():
    trader = Trader(style=EXTRAPOLATOR, params=EXTRAP_PARAMS)
    hist_rets = [0.05, 0.04, 0.03, 0.02, 0.01]
    adaptive_return = sum(hist_rets[i] * EXTRAP_PARAMS["weights"][i] for i in range(len(hist_rets)))
    squeeze_factor = EXTRAP_PARAMS["tanh"]["squeeze_factor"]
    max_dev = EXTRAP_PARAMS["tanh"]["max_dev"]
    central_return = EXTRAP_PARAMS["central_return"]
    exp_return = central_return + np.tanh((adaptive_return - central_return) / squeeze_factor) * max_dev
    assert trader.expected_return(hist_rets=hist_rets) == exp_return


def test_extrapolator_trader_expected_return_linear():
    params = EXTRAP_PARAMS
    params["use_tanh"] = False
    trader = Trader(style=EXTRAPOLATOR, params=params)
    hist_rets = [0.05, 0.04, 0.03, 0.02, 0.01]
    adaptive_return = sum(hist_rets[i] * EXTRAP_PARAMS["weights"][i] for i in range(len(hist_rets)))
    exp_return = EXTRAP_PARAMS["central_return"] + EXTRAP_PARAMS["linear"]["squeeze_factor"] * (
            adaptive_return - EXTRAP_PARAMS["central_return"])
    expected_return = max(min(exp_return, EXTRAP_PARAMS["linear"]["cap_return"]),
                          EXTRAP_PARAMS["linear"]["floor_return"])
    assert trader.expected_return(hist_rets=hist_rets) == expected_return
