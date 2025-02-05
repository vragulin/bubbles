""" Configuration file for the project. """

# Trader types
VALUE, EXTRAPOLATOR = range(2)

# Parameters for the exxtrapolator trader
DECAY_WEIGHTS = [0.36 * 0.75 ** i for i in range(5)]
EXTRAP_PARAMS = {
    "weights_unscaled": DECAY_WEIGHTS,  # Don't have to add up to 1, we will rescale them
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
        "max_dev": 0.05,
        "squeeze_factor": 0.20, # higher means lower slope at the center
        'floor_return': 0.0,
    }
}

# Simulation parameters
INIT_PRICE = 1
INIT_EARNINGS_ANN = 0.04
PAYOUT_RATIO = 0.5
INT_RATE = 0.00
SIM_MONTHS = 660
INIT_MONTHS = 60
