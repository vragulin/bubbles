import pytest as pt
from src.market_class import Trader, Market, VALUE, EXTRAPOLATOR
from src.sim import Simulation
from src.config import EXTRAP_PARAMS


def test_extrap_return():
    # Initialize the simulation
    sim = Simulation()
    price_list = [100, 95, 90, 85, 80, 75]
    hist_rets = [price_list[i] / price_list[i + 1] -1 for i in range(len(price_list) - 1)]
    # Set up the initial conditions
    month = 61  # Example month

    # Set up the price history
    for i, price in enumerate(price_list):
        sim.res.loc[month - 1 - 12 * i, 'price'] = price

    r_exp_expected= sim.market.traders[EXTRAPOLATOR].expected_return(hist_rets=hist_rets)
    r_exp_target, _ = sim.extrap_return(month)
    assert pt.approx(r_exp_expected) == r_exp_target
