""" Trader class to store information and functions related to the agents"""
import numpy as np
import copy
from src.config import VALUE, EXTRAPOLATOR


def merton_share(expected_return, risk_aversion, volatility):
    return expected_return / (risk_aversion * volatility ** 2)


class Trader:
    def __init__(self, style: int, cash: float = 0.5, shares: float = 0, gamma: float = 2, **kwargs):
        self.style = style
        self.cash = cash
        self.shares = shares
        self.gamma = gamma
        self.params = None
        self.prev_weight = None
        if self.style == EXTRAPOLATOR:
            self.params = copy.deepcopy(kwargs.get("params"))
            sum_weights = sum(self.params["weights_unscaled"])
            self.params["weights"] = [w / sum_weights for w in self.params["weights_unscaled"]]
            self.prev_weight = kwargs.get("prev_weight", None)
        else:
            self.params = None

    @staticmethod
    def merton_share(expected_return, risk_aversion, volatility):
        return expected_return / (risk_aversion * volatility ** 2)

    def expected_return(self, **kwargs):
        if self.style == VALUE:
            earnings, price = kwargs["earnings"], kwargs["price"]
            return self.value_expected_return(earnings, price)
        else:
            hist_rets = kwargs["hist_rets"]
            return self.extrapolator_expected_return(hist_rets)

    def code(self):
        return 'val' if self.style == VALUE else 'ext'

    @staticmethod
    def value_expected_return(earnings: float, price: float) -> float:
        """ Calculate the expected return for a value trader
        :param earnings: float, annualized earnings per share
        :return: float, expected return
        """

        return earnings / price

    def extrapolator_expected_return(self, hist_rets: list[float]):
        r_adaptive = sum(hist_rets[i] * self.params["weights"][i] for i in range(len(hist_rets)))
        r_central = self.params["central_return"]
        if self.params["use_tanh"]:
            squeeze_factor = self.params["tanh"]["squeeze_factor"]
            max_dev = self.params["tanh"]["max_dev"]
            r_exp = r_central + np.tanh((r_adaptive - r_central) / squeeze_factor) * max_dev
            if (floor_return := self.params["tanh"].get("floor_return")) is not None:
                r_exp = max(r_exp, floor_return)
        else:
            squeeze_factor = self.params["linear"]["squeeze_factor"]
            cap_return = self.params["linear"]["cap_return"]
            floor_return = self.params["linear"]["floor_return"]
            r_exp = r_central * squeeze_factor + r_adaptive * (1 - squeeze_factor)
            r_exp = min(max(r_exp, floor_return), cap_return)
        return r_exp

    def wealth(self, price):
        return self.cash + self.shares * price

    def equity_weight(self, price):
        return self.shares * price / self.wealth(price)

    def desired_eq_weight(self, sigma, **kwargs):
        if self.style == VALUE:
            expected_return = self.expected_return(earnings=kwargs['earnings'], price=kwargs['price'])
            return merton_share(expected_return, self.gamma, sigma)
        else:
            hist_rets = kwargs["hist_rets"]
            prev_weight = kwargs.get("prev_weight", self.prev_weight)
            expected_return = self.expected_return(hist_rets=hist_rets)
            long_term_tgt_weight = merton_share(expected_return, self.gamma, sigma)
            if prev_weight is None:
                return long_term_tgt_weight
            else:
                return prev_weight + self.params["adjustment_speed"] * (long_term_tgt_weight - prev_weight)

    def __repr__(self):
        return f"Trader(style={self.style}, cash={self.cash}, stock={self.shares}, gamma={self.gamma})"


class Market:
    # Describes the market on any given day
    def __init__(self, traders: list[Trader], earnings_ann: float, earnings_month: float,
                 payout_ratio: float, hist_rets: list[float], sigma: float):
        """ Initialize the market with a list of traders and the market parameters
        :param traders: list of Trader objects
        :param earnings_ann: float, annualized earnings per share (assume we reinvest monthly earnings at the current earnings yield
        :param monthly_earnings: float, monthly earnings per share - this is what drives dividends, growth, etc.
        :param payout_ratio: float, fraction of earnings paid out as dividends
        :param hist_rets: list of floats, historical returns, used by the extrapolator
        :param sigma: float, volatility of the market (exogenous)
        """
        self.traders = traders
        self.price = None
        self.earnings_ann = earnings_ann
        self.earnings_month = earnings_month
        self.payout_ratio = payout_ratio
        self.hist_rets = hist_rets
        self.sigma = sigma

    def __str__(self):
        return (f"Market(traders={self.traders}, earnings_ann={self.earnings_ann}, hist_rets={self.hist_rets}, "
                f"sigma={self.sigma}, price={self.price})")

    def pay_dividends(self):
        """
        Pay dividends to the traders, i.e. update
        """
        total_dividends = self.earnings_month * self.payout_ratio
        total_shares = sum(trader.shares for trader in self.traders)
        for trader in self.traders:
            trader.cash += trader.shares / total_shares * total_dividends

    def equilibrium_price(self):
        """
        Solve for the equilibrium price given the traders' strategies
        """
        # Set up a quadratic equation a* p^2 + b * p + c = 0 to solve for the price
        # Assume trader VALUE is the value trade, trader EXTRAPOLATOR is the extrapolator
        t_val, t_ext = self.traders[VALUE], self.traders[EXTRAPOLATOR]
        n_v, n_e = t_val.shares, t_ext.shares
        c_v, c_e = t_val.cash, t_ext.cash
        gamma_v = t_val.gamma
        s = self.sigma
        earn = self.earnings_ann
        omega_e = t_ext.desired_eq_weight(self.sigma, hist_rets=self.hist_rets, prev_weight=t_ext.prev_weight)

        # a = n_e * omega_e - 1
        a = n_e * omega_e - n_v - n_e  # UsE a more general formuls in case n_v + n_e != 1
        b = n_v * earn / gamma_v / s ** 2 + c_e * omega_e
        c = c_v * earn / gamma_v / s ** 2

        # Solve the quadratic equation
        if a > 0:
            p = (-b + np.sqrt(b ** 2 - 4 * a * c)) / 2 / a
        elif a < 0:
            p = (-b - np.sqrt(b ** 2 - 4 * a * c)) / 2 / a
        else:
            p = -c / b
        return p

    def imbalances(self) -> dict:
        """
        Calculate the difference between desired and actual holdings of cash and weights for each trader
        """
        imbalances = {}
        for trader in self.traders:
            eq_weight = trader.desired_eq_weight(self.sigma, earnings=self.earnings_ann, price=self.price,
                                                 hist_rets=self.hist_rets)
            imbalances[trader.style] = {
                "cash": trader.cash - trader.wealth(self.price) * (1 - eq_weight),
                "shares": (trader.equity_weight(self.price) - eq_weight) * trader.wealth(self.price) / self.price
            }
        return imbalances
