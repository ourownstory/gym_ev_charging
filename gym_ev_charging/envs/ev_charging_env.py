# from gym.utils import seeding
import datetime
from copy import deepcopy

import gym
import gym_utils as utils
import numpy as np
import pandas as pd
from data import toy_data
from gym import spaces
from gym_ev_charging.config_gym import get_config


# TODO create at proper location
# config = get_config('default')

class EVChargingEnv(gym.Env):
    """

    """
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        # TODO just init to zero
        pass

    def build(self, config=None):
        if config is None:
            config = get_config('default')
        self.config = config
        # TODO actually init
        self.num_stations = config.NUM_STATIONS
        self.episode_length = config.EPS_LEN
        self.time_step = config.TIME_STEP
        self.transformer_capacity = config.TRANSFORMER_CAPACITY
        self.reward_weights = config.REWARD_WEIGHTS  # completion, price, violation
        self.total_charging_data = utils.load_charging_data(config.path_data, config.NUM_STATIONS, config.TIME_STEP)
        # self.total_charging_data = config.TOTAL_CHARGING_DATA # dataframe, to be sampled from
        self.total_elec_price_data = pd.DataFrame()
        self.random_state = np.random.RandomState(config.RAND_SEED)
        self.observation_dimension = config.observation_dimension

        # gym stuff
        self.episode_over = False
        self.action_space = gym.spaces.Discrete(len(config.action_map))  # combination of decisions for all stations
        # self.observation_space = gym.spaces.Tuple(tuple([spaces.Discrete(2) for __ in range(82)]))
        self.observation_space = np.zeros(config.observation_dimension)
        self.total_steps = 0

        self.start_time = None
        self.charging_data = None
        self.elec_price_data = None
        self.durations = []
        self.done = False
        self.state = None
        self.reset()
        pass

        
    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        #translate action from number to tuple
        a = self.config.action_map[action]
        ob, reward = self.take_action(a) 
        episode_over = self.done
        return utils.featurize_s(ob), reward, episode_over, {}
    
    def charge_car(self, station, new_station, charge_rate):
        is_car, des_char, per_char, curr_dur =  station['is_car'], station['des_char'], station['per_char'], station['curr_dur']
        new_station['is_car'] = True
        new_station['des_char'] = des_char
        new_station['curr_dur'] = curr_dur + self.time_step
        curr_char = per_char*des_char
        new_station['char'] = min(des_char, curr_char +charge_rate*self.time_step)
        energy_added = new_station['char'] - curr_char
        new_station['per_char'] = float(new_station['char'])/des_char
        return energy_added
    
    def car_leaves(self, new_station):
        new_station['is_car'] = False
        new_station['des_char'], new_station['per_char'], new_station['dur'] = 0,0,0

    def car_arrives(self, new_station, session):
        new_station['is_car'] = True
        new_station['des_char'] = session[1]
        new_station['per_char'] = 0
        new_station['curr_dur'] = 0
    
    def take_action(self, a): 
        new_state = {}
        time = self.state['time']
        new_time = time + datetime.timedelta(hours=self.time_step)
        new_state['time'] = new_time
        stations, energy_charged, percent_charged = [],[],[]
        for stn_num, station in enumerate(self.state['stations']):
            new_station = deepcopy(station)
            if station['is_car']:
                energy_added = self.charge_car(station, new_station,a[stn_num])
                if new_station['curr_dur'] >= self.durations[stn_num]:
                    self.durations[stn_num] = 0
                    self.car_leaves(new_station)
            else:
                energy_added = 0
            #see if new car comes
            loc = self.charging_data[stn_num]
            next_start_time = datetime.datetime(9999, 1, 1) if len(loc) == 0 else loc[-1][0]
            if new_time >= next_start_time:
                #new car arrives
                session = loc.pop()
                self.durations[stn_num] = session[2]
                self.car_arrives(new_station, session)
            percent_charged.append(new_station['per_char'])
            energy_charged.append(energy_added)
            stations.append(new_station)
        new_state['stations'] = stations
        reward = self.reward(energy_charged, percent_charged, a)
        self.state = new_state
        self.done = sum([len(loc) for loc in self.charging_data]) + sum(self.durations) == 0
        return new_state, reward

    def get_initial_state(self):
        ## get_start_time
        initial_state = {}
        start_time = min([loc[-1][0] for loc in self.charging_data if len(loc) > 0])
        initial_state["time"] = start_time
        stations = []
        for loc in self.charging_data:
            station = {}
            if len(loc) >0 and (loc[-1][0] == start_time):
                ##process session
                session = loc.pop()
                self.durations.append(session[2])
                station["is_car"] = True
                station["des_char"] = session[1]
                station["per_char"] = 0
                station["curr_dur"] = 0 
            else:
                station["is_car"], station["des_char"], station["per_char"], station["curr_dur"] = False,0,0,0
                self.durations.append(0)
            stations.append(station)
        initial_state["stations"] = stations
        self.state = initial_state
        return initial_state

    # Called by take_action
    # the three arguments are lists of the given values at each station
    def reward(self, energy_charged, percent_charged, charging_powers):
        charge_reward = sum(np.array(energy_charged) * (np.exp(percent_charged) - 1))  # sum [0, energy_charged*(e-1)] (~[0, 8.5])

        elec_price = self.elec_price_data[self.get_current_state()['time'].to_pydatetime()]
        elec_cost = sum(np.array(energy_charged) * elec_price * (np.exp(1) - 1))  # sum [0, energy_charged*(e-1)] (~[0, 8.5])

        pow_violation = max(np.sum(charging_powers) - self.transformer_capacity, 0) / self.transformer_capacity
        pow_penalty = np.exp(pow_violation * 12) - 1  # [0, e^(10*pow_violation) - 1]  (~[0, 20])
        # print(charge_reward, elec_cost, pow_penalty)

        return self.reward_weights[0] * charge_reward - self.reward_weights[1] * elec_cost - self.reward_weights[2] * pow_penalty

    def get_current_state(self):
        return self.state

    def sample_data(self):
        # self.charging_data = deepcopy(locations)
        # TODO don't call toy_data here but self.total_elec_price_data
        self.elec_price_data = toy_data.price
        valid_dates = self.total_charging_data.loc[self.total_charging_data.index[0]:self.total_charging_data.index[-1] - datetime.timedelta(hours=self.episode_length * self.time_step)].index
        self.start_time = valid_dates[self.random_state.choice(range(len(valid_dates)))].to_pydatetime()
        self.charging_data = utils.sample_charging_data(self.total_charging_data, self.start_time, self.episode_length, self.time_step)

    def reset(self):
        self.done = False
        self.durations = []
        self.sample_data()
        self.state = self.get_initial_state()
        featurized_state = utils.featurize_s(self.state)
        return(featurized_state)

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        # return self.seed(seed)
        pass
