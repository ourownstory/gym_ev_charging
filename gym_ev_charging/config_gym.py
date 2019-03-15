import numpy as np
import itertools
import os


class config_default:
    def __init__(self):
        self.ENV_NAME = None
        self.RAND_SEED = 12345
        self.EPS_LEN = 4 * 24 * 7
        self.EVAL_EPS_LEN = 4*24*30

        self.NUM_STATIONS = 3
        self.TIME_STEP = 0.25
        self.MAX_POWER = 6.6
        self.MIN_POWER = 0.0
        self.NUM_POWER_STEPS = 3
        self.TRANSFORMER_LIMIT = 0.75
        self.TRANSFORMER_CAPACITY = self.MAX_POWER * self.NUM_STATIONS * self.TRANSFORMER_LIMIT
        self.REWARD_WEIGHTS = (0.333, 0.333, 0.333)

        self.charge_empty_first = True

        self.observation_dimension = 31 + 17*self.NUM_STATIONS
        self.actions = [np.linspace(self.MIN_POWER, self.MAX_POWER, self.NUM_POWER_STEPS)
                        for _ in range(self.NUM_STATIONS)]
        self.action_map = {idx: a for idx, a in enumerate(itertools.product(*self.actions))}
        # print(self.action_map)

        self.data_file = "sessions_161718_95014_top10.csv"
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path_data = os.path.join(cwd, "data", "clean", self.data_file)


class config_discrete(config_default):
    def __init__(self):
        super().__init__()
        self.ENV_NAME = "ev-charging-v0"


def get_config(config_name):
    if config_name == 'discrete':
        return config_discrete()

