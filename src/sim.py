""" Simulate Vic's 2-agent model of a stock market bubbles
    V. Ragulin - 01/25/2025
"""
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.market_class import Trader, Market
from src.config import *
import matplotlib.animation as animation

# Simulation parameters
# Trader Settings: [VALUE, EXTRAPOLATOR]
# INIT_CASH = [0.5, 0.5]
# INIT_SHARES = [0.5, 0.5]
INIT_FRACTION_VAL = 0.5  # Initial fraction of LT (Value) investors
RISK_AVERSION = [3, 3]  #
STK_VOL = 0.16
EARN_VOL = 0.1


class Simulation:

    def __init__(self):
        self.res = pd.DataFrame(columns=['month', 'price', 'total_ret_idx', 'fair_val',
                                         'earnings_ann', 'earnings_month',
                                         'dividends', 'reinvested',
                                         'total_cash', 'exp_ret_val', 'exp_ret_ext',
                                         'wealth_val', 'wealth_ext',
                                         'eq_weight_val', 'eq_weight_ext', 'dz'
                                         ])

        init_earn_monthly = ((1 + INIT_EARNINGS_ANN / INIT_PRICE) ** (1 / 12) - 1)
        self.market = Market(traders=[], earnings_ann=INIT_EARNINGS_ANN,
                             earnings_month=init_earn_monthly, payout_ratio=PAYOUT_RATIO,
                             hist_rets=[], sigma=STK_VOL)
        self.stk_vol = STK_VOL
        self.earn_vol = EARN_VOL
        self.rng = np.random.default_rng(123)
        self.initialize()

    def initialize(self):
        res = self.res
        res['month'] = np.arange(SIM_MONTHS + 1)
        res.set_index('month', inplace=True)

        # Total cash growht is exogenous
        # res['total_cash'] = sum(INIT_CASH) * (1 + INT_RATE) ** (np.arange(SIM_MONTHS + 1) / 12)
        res['dz'] = self.rng.normal(loc=0, scale=1, size=SIM_MONTHS + 1)

        # Initialize stock and eanings growth for the pre-trading period
        res.loc[0, ['price', 'total_ret_idx']] = INIT_PRICE
        for month in range(1, INIT_MONTHS + 1):
            if month == 1:
                res.loc[month, 'earnings_ann'] = INIT_EARNINGS_ANN
                res.loc[month, 'earnings_month'] = self.monthly_from_ann_earnings(INIT_EARNINGS_ANN, INIT_PRICE)
            else:
                curr_earnings_dict = self.this_month_earnings(month, vol=0)
                res.loc[month, 'earnings_ann'] = curr_earnings_dict['annual']
                res.loc[month, 'earnings_month'] = curr_earnings_dict['monthly']

            res.loc[month, 'dividends'] = res.loc[month, 'earnings_month'] * self.market.payout_ratio
            res.loc[month, 'reinvested'] = res.loc[month, 'earnings_month'] - res.loc[month, 'dividends']
            res.loc[month, 'price'] = res.loc[month - 1, 'price'] + res.loc[month, 'reinvested']
            res.loc[month, 'total_ret_idx'] = res.loc[month - 1, 'total_ret_idx'] \
                                              * (res.loc[month, 'price'] + res.loc[month, 'dividends']) \
                                              / res.loc[month - 1, 'price']

        # Initialize the traders
        wealth_share = [INIT_FRACTION_VAL, 1 - INIT_FRACTION_VAL]
        # Assume that both traders start with the same expected return
        init_exp_return = res.loc[INIT_MONTHS, 'earnings_ann'] / res.loc[INIT_MONTHS, 'price']
        merton_share = [Trader.merton_share(expected_return=init_exp_return, risk_aversion=RISK_AVERSION[t],
                                            volatility=STK_VOL) for t in [VALUE, EXTRAPOLATOR]]
        wm_tuples = list(zip(wealth_share, merton_share))
        starting_total_wealth = res.loc[INIT_MONTHS, 'price'] / sum(w * m for w, m in wm_tuples)

        starting_cash = [starting_total_wealth * w * (1 - m) for w, m in wm_tuples]
        res.loc[INIT_MONTHS, 'total_cash'] = sum(starting_cash)

        starting_price = res.loc[INIT_MONTHS, 'price']
        starting_shares = [starting_total_wealth * w * m / starting_price for w, m in wm_tuples]
        traders = []
        for s in [VALUE, EXTRAPOLATOR]:
            trader = Trader(style=s, cash=starting_cash[s], shares=starting_shares[s],
                            prev_weight=merton_share[s], gamma=RISK_AVERSION[s], params=EXTRAP_PARAMS)
            traders.append(trader)
            res.loc[INIT_MONTHS, 'wealth_' + trader.code()] = traders[s].wealth(price=starting_price)
            res.loc[INIT_MONTHS, 'eq_weight_' + trader.code()] = merton_share[s]
            res.loc[INIT_MONTHS, 'exp_ret_' + trader.code()] = init_exp_return

        self.market.traders = traders

    @staticmethod
    def ann_from_monthly_earnings(earnings_month: float, prev_price: float) -> float:
        """ Calculate annualized earnings from monthly earnings and previous price """
        return ((1 + earnings_month / prev_price) ** 12 - 1) * prev_price

    @staticmethod
    def monthly_from_ann_earnings(earnings_ann: float, prev_price: float) -> float:
        """ Calculate montly earnings from annualized earnings previous price """
        return ((1 + earnings_ann / prev_price) ** (1 / 12) - 1) * prev_price

    def this_month_earnings(self, month: int, vol: Optional[float] = None) -> dict:
        """ Calculate monthly and annualized earnings for the currrent period
        :param month: int, current month
        :param vol: float, optional, annualized earnings volatility, otherwise use the class attribute
        :return: dict, earnings for the next period, keys are 'monthly' and 'annual'
        """
        vol_used = vol if vol is not None else self.earn_vol
        earn_monthly = self.res.loc[month - 1, 'earnings_month'] \
                       * (1 + self.res.loc[month - 1, 'reinvested'] / self.res.loc[month - 1, 'price']) \
                       * (1 + vol_used * self.res.loc[month, 'dz'] / np.sqrt(12))
        earn_annual = self.ann_from_monthly_earnings(earn_monthly, self.res.loc[month - 1, 'price'])
        return {
            'monthly': earn_monthly,
            'annual': earn_annual
        }

    def extrap_return(self, month):
        """ Calculate the expected return for the extrapolator """
        res = self.res
        hist_rets = [
            res.loc[month - i * 12 - 1, 'total_ret_idx'] \
            / res.loc[month - i * 12 - 13, 'total_ret_idx'] - 1
            for i in range(5)
        ]
        return self.market.traders[EXTRAPOLATOR].expected_return(hist_rets=hist_rets), hist_rets

    def step(self, month: int):
        """ Simulate one step """

        res = self.res

        # Calculate the earnings for this period
        curr_earnings_dict = self.this_month_earnings(month)
        res.loc[month, 'earnings_ann'] = curr_earnings_dict['annual']
        res.loc[month, 'earnings_month'] = curr_earnings_dict['monthly']

        # Update the market structure and pay dividends
        self.market.earnings_ann = res.loc[month, 'earnings_ann']
        self.market.earnings_month = res.loc[month, 'earnings_month']
        self.market.pay_dividends()
        res.loc[month, 'dividends'] = self.market.earnings_month * self.market.payout_ratio
        res.loc[month, 'reinvested'] = self.market.earnings_month - res.loc[month, 'dividends']
        res.loc[month, 'total_cash'] = sum(self.market.traders[t].cash for t in [VALUE, EXTRAPOLATOR])

        # Solve for the market price
        res.loc[month, 'exp_ret_ext'], hist_rets = self.extrap_return(month)
        self.market.hist_rets = hist_rets
        price = self.market.equilibrium_price()
        traders = self.market.traders
        res.loc[month, 'price'] = price
        res.loc[month, 'exp_ret_val'] = traders[VALUE].expected_return(
            earnings=res.loc[month, 'earnings_ann'], price=price)
        res.loc[month, 'total_ret_idx'] = res.loc[month - 1, 'total_ret_idx'] \
                                          * (res.loc[month, 'price'] + res.loc[month, 'dividends']) \
                                          / res.loc[month - 1, 'price']
        for trader in traders:
            eq_weight = trader.desired_eq_weight(self.market.sigma,
                                                 earnings=self.market.earnings_ann,
                                                 price=price, hist_rets=hist_rets)
            w = trader.wealth(price=price)
            trader.shares = eq_weight * w / price
            trader.cash = w * (1 - eq_weight)
            trader.prev_weight = eq_weight
            res.loc[month, 'eq_weight_' + trader.code()] = trader.equity_weight(price=price)
            res.loc[month, 'wealth_' + trader.code()] = w


def animate_simulation1(sim):
    fig, ax = plt.subplots()
    ax.set_xlim(0, SIM_MONTHS)
    ax.set_ylim(sim.res['price'].min(), sim.res['price'].max())
    ax.set_xlabel('Month')
    ax.set_ylabel('Price')
    ax.set_title('Stock Price and Fair Value Over Time')

    line_price, = ax.plot([], [], label='Price')
    line_fair_val, = ax.plot([], [], label='Fair Value')
    ax.legend()

    def update(frame):
        line_price.set_data(sim.res.index[:frame], sim.res['price'][:frame])
        line_fair_val.set_data(sim.res.index[:frame], sim.res['fair_val'][:frame])
        return line_price, line_fair_val

    ani = animation.FuncAnimation(fig, update, frames=SIM_MONTHS + 1, interval=50, blit=True)
    ani.save('../results/stocks_sim1.gif', writer='pillow')
    plt.show()


def animate_simulation2(sim):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'hspace': 0.5})

    # Top plot: Stock Price and Fair Value
    ax1.set_xlim(0, SIM_MONTHS)
    ax1.set_ylim(sim.res['price'].min(), sim.res['price'].max() * 1.2)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Price')
    ax1.set_title('Stock Price and Fair Value Over Time')
    line_price, = ax1.plot([], [], label='Price', color='red')
    line_fair_val, = ax1.plot([], [], label='Fair Value', linestyle='--', color='blue')
    ax1.legend()

    # Bottom plot: Equity Weights
    ax2.set_xlim(0, SIM_MONTHS)
    ax2.set_ylim(0, 1.2)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Equity Weight')
    ax2.set_title('Equity Weights Over Time')
    line_eq_weight_val, = ax2.plot([], [], label='Value')
    line_eq_weight_ext, = ax2.plot([], [], label='Adaptive')
    ax2.legend()

    def update(frame):
        # Update top plot
        line_price.set_data(sim.res.index[:frame], sim.res['price'][:frame])
        line_fair_val.set_data(sim.res.index[:frame], sim.res['fair_val'][:frame])

        # Update bottom plot
        line_eq_weight_val.set_data(sim.res.index[:frame], sim.res['eq_weight_val'][:frame])
        line_eq_weight_ext.set_data(sim.res.index[:frame], sim.res['eq_weight_ext'][:frame])

        return line_price, line_fair_val, line_eq_weight_val, line_eq_weight_ext

    ani = animation.FuncAnimation(fig, update, frames=SIM_MONTHS + 1, interval=50, blit=True)
    ani.save('../results/stocks_sim2.gif', writer='pillow')
    plt.show()


def animate_simulation3(sim):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'hspace': 0.5})

    # Top plot: Stock Price and Fair Value
    ax1.set_xlim(0, SIM_MONTHS)
    ax1.set_ylim(sim.res['price'].min(), sim.res['price'].max() * 1.2)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Price')
    ax1.set_title('Stock Price and Fair Value Over Time')
    line_price, = ax1.plot([], [], label='Price', color='red')
    line_fair_val, = ax1.plot([], [], label='Fair Value', linestyle='--', color='blue')
    ax1.legend()

    # Middle plot: Equity Weights
    ax2.set_xlim(0, SIM_MONTHS)
    ax2.set_ylim(0, 1.2)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Equity Weight')
    ax2.set_title('Equity Weights Over Time')
    line_eq_weight_val, = ax2.plot([], [], label='Value')
    line_eq_weight_ext, = ax2.plot([], [], label='Adaptive')
    ax2.legend()

    # Bottom plot: Fraction of Wealth Value to Wealth Extrapolator
    ax3.set_xlim(0, SIM_MONTHS)
    wealth_frac = sim.res['wealth_val'] / (sim.res['wealth_val'] + sim.res['wealth_ext'])
    min_y = min(wealth_frac.min(), 1-wealth_frac.max())
    max_y = max(wealth_frac.max(), 1-wealth_frac.min())
    ax3.set_ylim(min_y * 0.9, max_y * 1.1)
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Fraction of Wealth')
    ax3.set_title('Share of Total Wealth Over Time - Value Investors')
    line_wealth_fraction, = ax3.plot([], [], label='Wealth% - Value', color='green')
    ax3.grid(True)
    ax3.legend()

    def update(frame):
        # Update top plot
        line_price.set_data(sim.res.index[:frame], sim.res['price'][:frame])
        line_fair_val.set_data(sim.res.index[:frame], sim.res['fair_val'][:frame])

        # Update middle plot
        line_eq_weight_val.set_data(sim.res.index[:frame], sim.res['eq_weight_val'][:frame])
        line_eq_weight_ext.set_data(sim.res.index[:frame], sim.res['eq_weight_ext'][:frame])

        # Update bottom plot
        wealth_fraction = sim.res['wealth_val'][:frame] / (
                sim.res['wealth_val'][:frame] + sim.res['wealth_ext'][:frame])
        line_wealth_fraction.set_data(sim.res.index[:frame], wealth_fraction)

        return line_price, line_fair_val, line_eq_weight_val, line_eq_weight_ext, line_wealth_fraction

    ani = animation.FuncAnimation(fig, update, frames=SIM_MONTHS + 1, interval=20, blit=True, repeat=False)
    ani.save('../results/stocks_sim3.gif', writer='pillow')
    plt.show()


def animate_phase2(sim):
    fig, ax = plt.subplots()
    ax.set_xlim(sim.res['wealth_val'].min(), sim.res['wealth_val'].max())
    ax.set_ylim(sim.res['wealth_ext'].min(), sim.res['wealth_ext'].max())
    ax.set_xlabel('Value Trader Wealth')
    ax.set_ylabel('Extrapolator Wealth')
    ax.set_title('Phase Space of Trader Wealths')

    # Use a continuous color palette
    norm = plt.Normalize(0, SIM_MONTHS)
    cmap = plt.get_cmap('viridis')
    sc = ax.scatter([], [], c=[], cmap=cmap, norm=norm)

    def update(frame):
        sc.set_offsets(np.c_[sim.res['wealth_val'][:frame], sim.res['wealth_ext'][:frame]])
        sc.set_array(np.arange(frame))
        return sc,

    ani = animation.FuncAnimation(fig, update, frames=SIM_MONTHS + 1, interval=50, blit=True)
    ani.save('../results/stocks_phase2.gif', writer='pillow')
    plt.show()


def main():
    sim = Simulation()

    for month in range(INIT_MONTHS + 1, SIM_MONTHS + 1):
        sim.step(month)

    # Calculate fair value - no need to do this in the loop
    sim.res['fair_val'] = sim.res['earnings_ann'] / EXTRAP_PARAMS['central_return']

    # Backfill columns for pre-trading period for better visualization
    pd.set_option('future.no_silent_downcasting', True)
    sim.res[['wealth_val', 'wealth_ext', 'eq_weight_val', 'eq_weight_ext']] = \
        sim.res[['wealth_val', 'wealth_ext', 'eq_weight_val', 'eq_weight_ext']
        ].bfill().infer_objects(copy=False)

    print(sim.res)
    sim.res[['price', 'fair_val']].plot(title="Stock Price and Fair Value")
    plt.show()

    # To Do - calculate final wealth of each trader, maybe add a chart
    animate_simulation3(sim)
    # animate_phase2(sim)


if __name__ == "__main__":
    main()
