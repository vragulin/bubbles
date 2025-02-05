import pytest as pt
import numpy as np
import copy
from src.market_class import Trader, merton_share
from src.config import VALUE, EXTRAPOLATOR

TEST_DECAY_WEIGHTS = [0.36 * 0.75 ** i for i in range(5)]
TEST_EXTRAP_PARAMS = {
    "weights_unscaled": TEST_DECAY_WEIGHTS,  # Don't have to add up to 1, we will rescale them
    "weights": [],
    "use_tanh": True,
    "central_return": 0.04,
    "adjustment_speed": 0.1,
    "linear": {
        "cap_return": 0.08,
        "floor_return": 0.01,
        "squeeze_factor": 0.5,
    },
    "tanh": {
        "max_dev": 0.04,
        "squeeze_factor": 0.10,  # higher means lower slope at the center
    }
}


@pt.fixture
def extrap_params():
    return copy.deepcopy(TEST_EXTRAP_PARAMS)


def test_trader_initialization():
    trader = Trader(style=VALUE, cash=1.0, shares=2.0, gamma=3.0)
    assert trader.style == VALUE
    assert trader.cash == 1.0
    assert trader.shares == 2.0
    assert trader.gamma == 3.0


def test_trader_repr(extrap_params):
    trader = Trader(style=EXTRAPOLATOR, cash=1.0, shares=2.0, gamma=3.0,
                    params=extrap_params)
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


def test_extrapolator_trader_expected_return_tanh(extrap_params):
    trader = Trader(style=EXTRAPOLATOR, params=extrap_params)
    hist_rets = [0.05, 0.04, 0.03, 0.02, 0.01]
    adaptive_return = sum(hist_rets[i] * trader.params["weights"][i] for i in range(len(hist_rets)))
    squeeze_factor = extrap_params["tanh"]["squeeze_factor"]
    max_dev = extrap_params["tanh"]["max_dev"]
    central_return = extrap_params["central_return"]
    exp_return = central_return + np.tanh((adaptive_return - central_return) / squeeze_factor) * max_dev
    assert trader.expected_return(hist_rets=hist_rets) == exp_return


def test_extrapolator_trader_expected_return_linear(extrap_params):
    p = extrap_params  # alias
    p["use_tanh"] = False
    trader = Trader(style=EXTRAPOLATOR, params=p)
    hist_rets = [0.05, 0.04, 0.03, 0.02, 0.01]
    adaptive_return = sum(hist_rets[i] * trader.params["weights"][i] for i in range(len(hist_rets)))
    central_ret = p["central_return"]
    exp_return = central_ret + p["linear"]["squeeze_factor"] * (adaptive_return - central_ret)
    expected_return = max(min(exp_return, p["linear"]["cap_return"]),
                          p["linear"]["floor_return"])
    assert trader.expected_return(hist_rets=hist_rets) == expected_return


def test_desired_eq_weight(extrap_params):
    p = extrap_params  # alias
    p["use_tanh"] = True
    trader = Trader(style=EXTRAPOLATOR, params=p, gamma=2.0)
    hist_rets = [0.05, 0.04, 0.03, 0.02, 0.01]
    prev_weight = 0.8
    sigma = 0.2
    adaptive_return = sum(hist_rets[i] * trader.params["weights"][i] for i in range(len(hist_rets)))
    squeeze_factor = p["tanh"]["squeeze_factor"]
    max_dev = p["tanh"]["max_dev"]
    central_return = p["central_return"]
    alpha = p["adjustment_speed"]
    exp_return = central_return + np.tanh((adaptive_return - central_return) / squeeze_factor) * max_dev
    lt_weight_target = merton_share(exp_return, trader.gamma, 0.2)
    expected_desired_weight = (1 - alpha) * prev_weight + alpha * lt_weight_target
    desired_weight = trader.desired_eq_weight(sigma, hist_rets=hist_rets, prev_weight=prev_weight)
    assert desired_weight == expected_desired_weight


def test_desired_eq_weight_with_prev_weight_in_class(extrap_params):
    p = extrap_params  # alias
    p["use_tanh"] = True
    trader = Trader(style=EXTRAPOLATOR, params=p, gamma=2.0)
    trader.prev_weight = 0.8
    hist_rets = [0.05, 0.04, 0.03, 0.02, 0.01]
    sigma = 0.2
    adaptive_return = sum(hist_rets[i] * trader.params["weights"][i] for i in range(len(hist_rets)))
    squeeze_factor = p["tanh"]["squeeze_factor"]
    max_dev = p["tanh"]["max_dev"]
    central_return = p["central_return"]
    alpha = p["adjustment_speed"]
    exp_return = central_return + np.tanh((adaptive_return - central_return) / squeeze_factor) * max_dev
    lt_weight_target = merton_share(exp_return, trader.gamma, sigma)
    expected_desired_weight = (1 - alpha) * trader.prev_weight + alpha * lt_weight_target
    desired_weight = trader.desired_eq_weight(sigma, hist_rets=hist_rets)
    assert desired_weight == expected_desired_weight


def test_desired_eq_weight_with_none_prev_weight(extrap_params):
    p = extrap_params  # alias
    p["use_tanh"] = True
    trader = Trader(style=EXTRAPOLATOR, params=p, gamma=2.0)
    hist_rets = [0.05, 0.04, 0.03, 0.02, 0.01]
    sigma = 0.2
    adaptive_return = sum(hist_rets[i] * trader.params["weights"][i] for i in range(len(hist_rets)))
    squeeze_factor = p["tanh"]["squeeze_factor"]
    max_dev = p["tanh"]["max_dev"]
    central_return = p["central_return"]
    exp_return = central_return + np.tanh((adaptive_return - central_return) / squeeze_factor) * max_dev
    lt_weight_target = merton_share(exp_return, trader.gamma, sigma)
    desired_weight = trader.desired_eq_weight(sigma, hist_rets=hist_rets, prev_weight=None)
    assert desired_weight == lt_weight_target
    desired_weight = trader.desired_eq_weight(sigma, hist_rets=hist_rets)
    assert desired_weight == lt_weight_target
