

class config_default:
    def __init__(self):
        self.ENV_NAME = None
        self.continuous_actions = None
        self.discretize_obs = None
        self.do_not_featurize = False
        self.alt_reward_func = False
        self.will_reward_func = False

        self.RAND_SEED = 12345
        self.TIME_STEP = 0.25
        self.MAX_POWER = 6.6
        self.MIN_POWER = 0.0
        self.EPS_LEN = 4*24*3
        self.EVAL_EPS_LEN = 4*24*3

        self.NUM_STATIONS = 1
        self.TRANSFORMER_LIMIT = 1  # [0, 1]
        self.solar_behind_meter = 0  # [0, (1 - TRANSFORMER_LIMIT) / TRANSFORMER_LIMIT]
        self.charge_empty_factor = 0  # [0, 2]

        self.REWARD_WEIGHTS = (1, 0, 0)
        self.penalize_unecessary_actions = 0

        self.train_file = "train_sessions_161718_95014_top10.csv"
        self.eval_file = "eval_sessions_161718_95014_top10.csv"


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
        self.alt_reward_func = True
        self.RAND_SEED = 1
        self.penalize_unecessary_actions = 0
        self.NUM_STATIONS = 1
        self.NUM_POWER_STEPS = 2  # [2, 10]
        self.TRANSFORMER_LIMIT = 1.0
        self.EPS_LEN = 4 * 24 * 3
        self.EVAL_EPS_LEN = 4 * 24 * 3
        self.REWARD_WEIGHTS = (0.55, 0.45, 0)

class config_will(config_discrete):
    def __init__(self):
        super().__init__()
        #self.do_not_featurize = True
        self.will_reward_func= True
        self.RAND_SEED = 1
        #self.penalize_unecessary_actions = 0
        self.NUM_STATIONS = 1
        self.NUM_POWER_STEPS = 2  # [2, 10]
        self.TRANSFORMER_LIMIT = 1.0
        self.EPS_LEN = 4 * 24 * 3
        self.EVAL_EPS_LEN = 4 * 24 * 3
        self.charge_weight = 0.51
        self.REWARD_WEIGHTS = (self.charge_weight, 1-self.charge_weight, 0)
        self.alpha = 1
        self.featurize_will = True

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
    if config_name == 'will':
        return config_will()

