from gym_utils import *
import os
EPS_LEN = 4 * 24 * 7
NUM_STATIONS = 3
TIME_STEP = 0.25
TRANSFORMER_CAPACITY = 6.6 * NUM_STATIONS * 0.80
REWARD_WEIGHTS = (0.333, 0.333, 0.333)
TOTAL_CHARGING_DATA = load_charging_data(os.path.join("data", "clean", "sessions_161718_95014_top10.csv"), NUM_STATIONS, TIME_STEP)
RAND_SEED = 100
