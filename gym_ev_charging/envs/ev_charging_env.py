# from gym.utils import seeding
import datetime
from copy import deepcopy
import numpy as np
import itertools
import pandas as pd
from gym import spaces
from gym_ev_charging.config_gym import get_config


import gym
import gym_utils as utils
from data import toy_data

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
        self.num_stations = config.NUM_STATIONS
        self.episode_length = config.EPS_LEN
        self.time_step = config.TIME_STEP
        self.max_power = config.MAX_POWER
        self.min_power = config.MIN_POWER
        self.transformer_capacity = config.MAX_POWER * config.NUM_STATIONS * config.TRANSFORMER_LIMIT
        self.reward_weights = config.REWARD_WEIGHTS  # completion, price, violation
        self.total_charging_data = utils.load_charging_data(config.path_data, config.NUM_STATIONS, config.TIME_STEP)
        self.total_elec_price_data = pd.DataFrame()
        self.random_state = np.random.RandomState(config.RAND_SEED)

        if config.discretize_obs:
            self.featurize = utils.featurize_s
            self.observation_dimension = 24 + 7 + 17*config.NUM_STATIONS
        else:
            self.featurize = utils.featurize_cont
            self.observation_dimension = 1 + 7 + 4*config.NUM_STATIONS
        self.observation_space = np.zeros(self.observation_dimension)

        if self.config.continuous_actions:
            self.action_space = np.zeros(config.NUM_STATIONS)
        else:
            actions = [np.linspace(config.MIN_POWER, config.MAX_POWER, config.NUM_POWER_STEPS)
                            for _ in range(config.NUM_STATIONS)]
            # combination of decisions for all stations
            self.action_map = {idx: np.array(a) for idx, a in enumerate(itertools.product(*actions))}
            self.action_space = gym.spaces.Discrete(len(self.action_map))

        self.info = None

        self.total_steps = 0
        self.episode_over = False
        self.evaluation_mode = False

        self.charging_data = None
        self.elec_price_data = None
        self.durations = []
        self.done = False
        self.state = None

        # get initial state
        self.reset()
        
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
        self.info = {
            'new_state': None,
            'charge_rates': None,
            'elec_cost': None,
            'finished_cars_stats': []
        }
        if not self.config.continuous_actions:
            #translate action from number to tuple
            action = self.action_map[action]
        action = utils.scale_action(action, self.transformer_capacity)
        new_state, reward = self.take_action(action)
        #translate action from number to tuple
        episode_over = self.done
        self.info['new_state'], self.info['charge_rates'] = new_state, action
        return self.featurize(new_state), reward, episode_over, self.info
    
    def charge_car(self, station, new_station, charge_rate):
        is_car, des_char, per_char, curr_dur =  station['is_car'], station['des_char'], station['per_char'], station['curr_dur']
        new_station['is_car'] = True
        new_station['des_char'] = des_char
        new_station['curr_dur'] = curr_dur + self.time_step
        curr_char = per_char*des_char
        total_char = min(des_char, curr_char +charge_rate*self.time_step)
        energy_added = total_char - curr_char
        new_station['per_char'] = float(total_char)/des_char
        return energy_added
    
    def car_leaves(self, new_station):
        #compute statistics for self.info
        total_char = new_station['des_char']*new_station['per_char']
        best_possible_char = min(new_station['curr_dur']*self.max_power, new_station['des_char'])
        self.info['finished_cars_stats'].append(total_char/best_possible_char)
        #reset the station
        new_station['is_car'] = False
        new_station['des_char'], new_station['per_char'], new_station['curr_dur'] = 0,0,0

    def car_arrives(self, new_station, session):
        new_station['is_car'] = True
        new_station['des_char'] = session[1]
        new_station['per_char'] = 0
        new_station['curr_dur'] = 0
    
    def take_action(self, a):
        # print(self.state)
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
        reward = self.reward(
            energy_charged=energy_charged,
            percent_charged=percent_charged
        )
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
    def reward_exp(self, energy_charged, percent_charged, charging_powers):
        charge_reward = sum(np.array(energy_charged) * (np.exp(percent_charged) - 1))  # sum [0, energy_charged*(e-1)] (~[0, 8.5])

        elec_price = self.elec_price_data[self.get_current_state()['time'].to_pydatetime()]
        elec_cost = sum(np.array(energy_charged) * elec_price * (np.exp(1) - 1))  # sum [0, energy_charged*(e-1)] (~[0, 8.5])

        pow_violation = max(np.sum(charging_powers) - self.transformer_capacity, 0) / self.transformer_capacity
        pow_penalty = np.exp(pow_violation * 12) - 1  # [0, e^(10*pow_violation) - 1]  (~[0, 20])
        # print(charge_reward, elec_cost, pow_penalty)

        return self.reward_weights[0] * charge_reward - self.reward_weights[1] * elec_cost - self.reward_weights[2] * pow_penalty

    # Called by take_action
    def reward(self, energy_charged, percent_charged):
        if self.config.charge_empty_factor > 0:
            charge_influence = 1.0 + self.config.charge_empty_factor * (0.5 - np.array(percent_charged))
        else:
            charge_influence = 1

        charge_reward = np.sum(energy_charged * charge_influence)
        charge_reward = 1000 * charge_reward / (self.num_stations*self.time_step*self.max_power)  # [0, 1000]
        # print("percent_charged", percent_charged)
        # print("charge_influence", charge_influence)
        # print("energy_charged", energy_charged)
        # print("energy_charged * charge_influence", energy_charged * charge_influence)
        # print("charge_reward", charge_reward)

        elec_price = self.elec_price_data[self.get_current_state()['time'].to_pydatetime()]
        elec_cost = np.sum(energy_charged) * elec_price
        #store statistics
        self.info['elec_cost'] = elec_cost
        elec_cost = 1000 * elec_cost / (self.num_stations*self.time_step*self.max_power)  # [0, 1000] if price [0,1]
        capa = self.transformer_capacity
        if self.config.solar_behind_meter > 0:
            # with elec_price as inverse of solar output, increase transformer capa relative to solar generation
            capa = capa * (1 + self.config.solar_behind_meter * (1 - elec_price))

        pow_violation = (np.sum(energy_charged) / self.time_step) - capa
        if pow_violation > 0:
            pow_ratio = min(1, pow_violation / capa)
            pow_penalty = np.exp(np.log(1000)*pow_ratio) - 1.0  # [0, 1000]
        else:
            pow_penalty = 0.0
        assert abs(pow_penalty) < 1e-3

        reward = [charge_reward, -elec_cost, -pow_penalty]
        # print("np.sum(energy_charged, self.transformer_capacity", np.sum(energy_charged), self.transformer_capacity)
        # print("energy_charged", energy_charged)
        # print("percent_charged", percent_charged)
        # print("reward", reward)
        reward = sum([r*w for r, w in zip(reward, self.reward_weights)]) / sum(self.reward_weights)
        return reward

    def get_current_state(self):
        return self.state

    def sample_data(self):
        # TODO don't call toy_data here but self.total_elec_price_data
        elec_price_data = toy_data.price

        if self.evaluation_mode:
            charging_data = utils.sample_charging_data(
                self.total_charging_data,
                self.config.EVAL_EPS_LEN,
                self.time_step,
                self.random_state
            )
        else:
            charging_data = utils.sample_charging_data(
                self.total_charging_data,
                self.episode_length,
                self.time_step,
                self.random_state
            )
        return charging_data, elec_price_data

    def reset(self):
        self.done = False
        self.info = None
        self.durations = []
        self.charging_data, self.elec_price_data = self.sample_data()
        self.state = self.get_initial_state()
        featurized_state = self.featurize(self.state)
        return featurized_state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        # return self.seed(seed)
        pass
