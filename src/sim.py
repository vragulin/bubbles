""" Simulate Vic's 2-agent model of a stock market bubbles
    V. Ragulin - 01/25/2025
"""
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
INIT_FRACTION_VAL = 0.6  # Initial fraction of LT (Value) investors
RISK_AVERSION = [2, 3]  #
STK_VOL = 0.16
EARNINGS_PARAMS = {
    "mean_growth": 0.04,
    "vol": 0.1
}


class Simulation:

    def __init__(self):
        self.res = pd.DataFrame(columns=['month', 'price', 'fair_val', 'earnings', 'total_cash',
                                         'exp_ret_val', 'exp_ret_ext',
                                         'wealth_val', 'wealth_ext',
                                         'eq_weight_val', 'eq_weight_ext', 'dz'
                                         ])

        self.market = Market(traders=[], earnings_curr=INIT_EARNINGS,
                             hist_rets=[], sigma=STK_VOL)
        self.stk_vol = STK_VOL
        self.earnings_params = EARNINGS_PARAMS
        self.rng = np.random.default_rng(123)
        self.initialize()

    def initialize(self):
        res = self.res
        res['month'] = np.arange(SIM_MONTHS + 1)
        res.set_index('month', inplace=True)

        # Total cash growht is exogenous
        # res['total_cash'] = sum(INIT_CASH) * (1 + INT_RATE) ** (np.arange(SIM_MONTHS + 1) / 12)
        res['dz'] = self.rng.normal(loc=0, scale=1, size=SIM_MONTHS + 1)

        # Initialize stock and eanings grwoth for the initial period
        res.iloc[:INIT_MONTHS + 1, res.columns.get_loc("earnings")] \
            = INIT_EARNINGS * (1 + EARNINGS_PARAMS['mean_growth']) ** (np.arange(INIT_MONTHS + 1) / 12)
        res.iloc[:INIT_MONTHS + 1, res.columns.get_loc("price")] \
            = INIT_PRICE * (1 + EARNINGS_PARAMS['mean_growth']) ** (np.arange(INIT_MONTHS + 1) / 12)

        # Initialize the traders
        wealth_share = [INIT_FRACTION_VAL, 1 - INIT_FRACTION_VAL]
        merton_share = [Trader.merton_share(expected_return=EARNINGS_PARAMS['mean_growth'],
                                            risk_aversion=RISK_AVERSION[t], volatility=STK_VOL)
                        for t in [VALUE, EXTRAPOLATOR]]
        wm_tuples = list(zip(wealth_share, merton_share))
        starting_total_wealth = res.loc[INIT_MONTHS, 'price'] / sum(w * m for w, m in wm_tuples)
        starting_cash = [starting_total_wealth * w * (1 - m) for w, m in wm_tuples]
        starting_price = res.loc[INIT_MONTHS, 'price']
        starting_shares = [starting_total_wealth * w * m / starting_price for w, m in wm_tuples]
        traders = []
        for s in [VALUE, EXTRAPOLATOR]:
            trader = Trader(style=s, cash=starting_cash[s], shares=starting_shares[s],
                            gamma=RISK_AVERSION[s], params=EXTRAP_PARAMS)
            traders.append(trader)
            res.loc[INIT_MONTHS, 'wealth_' + trader.code()] = traders[s].wealth(price=starting_price)
            res.loc[INIT_MONTHS, 'eq_weight_' + trader.code()] = merton_share[s]
            res.loc[INIT_MONTHS, 'exp_ret_' + trader.code()] = EARNINGS_PARAMS['mean_growth']

        self.market.traders = traders

    def extrap_return(self, month):
        """ Calculate the expected return for the extrapolator """
        res = self.res
        hist_rets = [
            res.loc[month - i * 12 - 1, 'price'] \
            / res.loc[month - i * 12 - 13, 'price'] - 1
            for i in range(5)
        ]
        return self.market.traders[EXTRAPOLATOR].expected_return(hist_rets=hist_rets), hist_rets

    def step(self, month):
        """ Simulate one step """

        res = self.res
        # Calculate the earnings for the next period
        res.loc[month, 'earnings'] = res.loc[month - 1, 'earnings'] * (
                (1 + res.loc[month - 1, 'earnings'] / res.loc[month - 1, 'price']) *
                (1 + self.earnings_params['vol'] * res.loc[month, 'dz'])
        ) ** (1 / 12)

        res.loc[month, 'exp_ret_ext'], hist_rets = self.extrap_return(month)

        # Solve for the market price
        self.market.earnings_curr = res.loc[month, 'earnings']
        self.market.hist_rets = hist_rets
        price = self.market.equilibrium_price()
        traders = self.market.traders
        res.loc[month, 'price'] = price
        res.loc[month, 'exp_ret_val'] = traders[VALUE].expected_return(
            earnings=res.loc[month, 'earnings'], price=price)
        for trader in traders:
            eq_weight = trader.desired_eq_weight(self.market.sigma,
                                                 earnings=self.market.earnings_curr,
                                                 price=price, hist_rets=hist_rets)
            w = trader.wealth(price=price)
            trader.shares = eq_weight * w / price
            trader.cash = w * (1 - eq_weight)
            res.loc[month, 'eq_weight_' + trader.code()] = trader.equity_weight(price=price)
            res.loc[month, 'wealth_' + trader.code()] = w

        res['fair_val'] = res['earnings'] / INIT_PRICE / INIT_EARNINGS


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
    ax3.set_ylim(0.5, 0.7)
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

    ani = animation.FuncAnimation(fig, update, frames=SIM_MONTHS + 1, interval=50, blit=True, repeat=False)
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

    pd.set_option('future.no_silent_downcasting', True)
    sim.res[['wealth_val', 'wealth_ext', 'eq_weight_val', 'eq_weight_ext']] = \
        sim.res[['wealth_val', 'wealth_ext', 'eq_weight_val', 'eq_weight_ext']
        ].bfill().infer_objects(copy=False)

    print(sim.res)
    sim.res[['price', 'fair_val']].plot(title="Stock Price and Fair Value")
    plt.show()

    # To Do - calculate final wealth of each trader, maybe add a chart
    # animate_simulation3(sim)
    animate_phase2(sim)


if __name__ == "__main__":
    main()
