import os


class config_default:
    def __init__(self):
        self.ENV_NAME = None
        self.continuous_actions = None
        self.discretize_obs = None
        self.do_not_featurize = False

        self.RAND_SEED = 123
        self.TIME_STEP = 0.25
        self.MAX_POWER = 6.6
        self.MIN_POWER = 0.0
        self.EPS_LEN = 4*24*7
        self.EVAL_EPS_LEN = 4*24*7

        self.NUM_STATIONS = 3
        self.TRANSFORMER_LIMIT = 0.33  # [0, 1]
        self.solar_behind_meter = 0  # [0, (1 - TRANSFORMER_LIMIT) / TRANSFORMER_LIMIT]
        self.charge_empty_factor = 0  # [0, 2]

        self.REWARD_WEIGHTS = (1, 0, 0)
        self.penalize_unecessary_actions = 100

        self.data_file = "sessions_161718_95014_top10.csv"
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path_data = os.path.join(cwd, "data", "clean", self.data_file)


class config_discrete(config_default):
    def __init__(self):
        super().__init__()
        self.ENV_NAME = "ev-charging-v0"
        self.continuous_actions = False
        self.discretize_obs = True
        self.NUM_POWER_STEPS = 2  # [2, 10]


class config_dc(config_discrete):
    def __init__(self):
        super().__init__()
        self.discretize_obs = False


class config_cont(config_default):
    def __init__(self):
        super().__init__()
        self.ENV_NAME = "ev-charging-v0"
        self.continuous_actions = True
        self.discretize_obs = False


class config_cd(config_cont):
    def __init__(self):
        super().__init__()
        self.discretize_obs = True

class config_justin(config_dc):
    def __init__(self):
        super().__init__()
        self.do_not_featurize = True
        self.NUM_STATIONS = 3
        self.NUM_POWER_STEPS = 3  # [2, 10]
        self.TRANSFORMER_LIMIT = 0.7
        self.EPS_LEN = 4 * 24 * 3
        self.EVAL_EPS_LEN = 4 * 24 * 3

def get_config(config_name):
    if config_name == 'discrete':
        return config_discrete()
    if config_name == 'cont':
        return config_cont()
    if config_name == 'DC':
        return config_dc()
    if config_name == 'CD':
        return config_cd()
    if config_name == 'justin':
        return config_justin()

