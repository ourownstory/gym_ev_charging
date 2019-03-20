

def get_config(name):
    return globals()[name]()


class Config:
    def __init__(self):
        self.ENV_NAME = None
        self.continuous_actions = None
        self.obs_features = "combined"  # "discrete", "continuous", "combined"
        self.do_not_featurize = False
        self.alt_reward_func = False

        self.use_delayed_charge_reward = False  #'leave', 'full', 'full_bonus'
        self.reward_magnitude = 1.0
        self.RAND_SEED = 12345
        self.TIME_STEP = 0.25
        self.MAX_POWER = 6.6
        self.MIN_POWER = 0.0

        self.EPS_LEN = 4*24*3
        self.EVAL_EPS_LEN = self.EPS_LEN

        self.NUM_STATIONS = 1
        self.TRANSFORMER_LIMIT = 1  # [0, 1]
        self.scale_actions_transformer = True
        self.solar_behind_meter = 0  # [0, (1 - TRANSFORMER_LIMIT) / TRANSFORMER_LIMIT]
        self.charge_empty_factor = 0  # [0, 2]

        self.REWARD_WEIGHTS = (1, 1, 0)
        self.penalize_unecessary_actions = 0
        self.end_after_leave = True  # only affects when training on single station

        self.train_file = "train_sessions_161718_95014_top10.csv"
        self.eval_file = "eval_sessions_161718_95014_top10.csv"


class Discrete(Config):
    def __init__(self):
        super().__init__()
        self.ENV_NAME = "ev-charging-v0"
        # self.obs_features = "discrete"  # default combined
        self.discretize_obs = True
        self.NUM_POWER_STEPS = 2  # [2, 10]


class DC(Discrete):
    def __init__(self):
        super().__init__()
        self.obs_features = "continuous"


class Continuous(Config):
    def __init__(self):
        super().__init__()
        self.ENV_NAME = "ev-charging-v0"
        self.continuous_actions = True
        # self.obs_features = "continuous"   # default combined


class CD(Continuous):
    def __init__(self):
        super().__init__()
        self.obs_features = "discrete"


class Justin(DC):
    def __init__(self):
        super().__init__()
        self.do_not_featurize = True
        self.alt_reward_func = True
        self.use_delayed_charge_reward = "full_bonus" #False "leave" "full" "full_bonus"

        self.RAND_SEED = 1
        self.NUM_STATIONS = 2
        self.NUM_POWER_STEPS = 2  # [2, 10]
        self.TRANSFORMER_LIMIT = 1.0
        self.EPS_LEN = 4 * 24 * 3
        self.EVAL_EPS_LEN = 4 * 24 * 3
        self.REWARD_WEIGHTS = (0.55, 0.45, 0)

        self.penalize_unecessary_actions = 0
        self.reward_magnitude = 1.0


class Single(Discrete):
    def __init__(self):
        super().__init__()
        self.use_delayed_charge_reward = "full"  #'leave', 'full', 'full_bonus'

        self.EPS_LEN = 4*24
        self.EVAL_EPS_LEN = self.EPS_LEN

        self.NUM_STATIONS = 1
        self.TRANSFORMER_LIMIT = 1  # [0, 1]
        self.scale_actions_transformer = True

        self.solar_behind_meter = 0  # [0, (1 - TRANSFORMER_LIMIT) / TRANSFORMER_LIMIT]
        self.charge_empty_factor = 0  # [0, 2]

        self.REWARD_WEIGHTS = (1, 1, 0)

        self.penalize_unecessary_actions = 0
        self.end_after_leave = True  # only affects when training on single station
