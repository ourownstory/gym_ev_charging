import numpy as np
import itertools
import os


class config_default:
    def __init__(self):
        self.env_name = None
        self.EPS_LEN = 4 * 24 * 7
        self.NUM_STATIONS = 3
        self.TIME_STEP = 0.25
        self.TRANSFORMER_CAPACITY = 6.6 * self.NUM_STATIONS * 0.80
        self.REWARD_WEIGHTS = (0.333, 0.333, 0.333)
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path_data = os.path.join(cwd, "data", "clean", "sessions_161718_95014_top10.csv")
        # self.TOTAL_CHARGING_DATA = load_charging_data(path_data, NUM_STATIONS, TIME_STEP)
        self.RAND_SEED = 100
        self.max_power = [6.6 for _ in range(self.NUM_STATIONS)]
        self.min_power = [0 for _ in range(self.NUM_STATIONS)]
        self.num_power_steps = [2 for _ in range(self.NUM_STATIONS)]
        self.actions = [np.linspace(self.min_power[i], self.max_power[i], self.num_power_steps[i])
                        for i in range(self.NUM_STATIONS)]
        self.action_map = {idx: a for idx, a in enumerate(itertools.product(*self.actions))}
        self.observation_dimension = 82


class config_discrete(config_default):
    def __init__(self):
        super().__init__()
        self.env_name = "ev-charging-v0"


def get_config(config_name):
    if config_name == 'discrete':
        return config_discrete()

