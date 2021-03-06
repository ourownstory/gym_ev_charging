# from gym.utils import seeding
import datetime
from copy import deepcopy
import numpy as np
import itertools
import pandas as pd
from gym import spaces
from gym_ev_charging.config_gym import get_config
import os

import gym
import gym_utils as utils
from data import toy_data


class EVChargingEnv(gym.Env):
    """EVChargingEnv implements an OpenAI Gym Environment that simulates car arrivals, departures, and charging
    decisions from real data.

    """
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        self.info = None

        self.evaluation_mode = False

        self.charging_data = None
        self.elec_price_data = None
        self.durations = []
        self.done = False
        self.state = None

    def build(self, config=None):
        """
        Builds the environment based on the settings in config
        :param config:
        :return:
        """
        if config is None:
            config = get_config('default')
        self.config = config

        self.random_state = np.random.RandomState(config.RAND_SEED)

        self.reward_range = (-config.reward_magnitude, config.reward_magnitude)
        self.num_stations = config.NUM_STATIONS
        self.episode_length = config.EPS_LEN
        self.time_step = config.TIME_STEP
        self.max_power = config.MAX_POWER
        self.min_power = config.MIN_POWER
        self.transformer_capacity = config.MAX_POWER * config.NUM_STATIONS * config.TRANSFORMER_LIMIT
        # completion, price, violation
        self.reward_weights = [x / float(sum(config.REWARD_WEIGHTS)) for x in config.REWARD_WEIGHTS]
        self.delayed_charge_reward = [0] * self.num_stations

        cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(cwd, "data", "clean")
        train_data = os.path.join(path, self.config.train_file)
        eval_data = os.path.join(path, self.config.eval_file)
        self.train_charging_data = utils.load_charging_data(train_data, config.NUM_STATIONS, config.TIME_STEP)
        self.eval_charging_data = utils.load_charging_data(eval_data, config.NUM_STATIONS, config.TIME_STEP)
        self.total_elec_price_data = pd.DataFrame()

        if config.obs_features == 'discrete':
            self.featurize = utils.featurize_s
            self.observation_dimension = 24+7+5 + 17*config.NUM_STATIONS
        elif config.obs_features == 'continuous':
            self.featurize = utils.featurize_cont
            self.observation_dimension = 4 + 5*config.NUM_STATIONS
            if config.do_not_featurize:
                self.featurize = lambda x: x
                self.observation_dimension = 7 + 3 + 5 * config.NUM_STATIONS
        elif config.obs_features == 'combined':
            self.featurize = utils.featurize_comb
            self.observation_dimension = 24+7+5 + 4 + (17+5)*config.NUM_STATIONS

        self.observation_space = np.zeros(self.observation_dimension)

        if self.config.continuous_actions:
            self.action_space = np.zeros(config.NUM_STATIONS)
        else:
            self.actions = [np.linspace(config.MIN_POWER, config.MAX_POWER, config.NUM_POWER_STEPS)
                            for _ in range(config.NUM_STATIONS)]
            # combination of decisions for all stations
            self.action_map = {idx: np.array(a) for idx, a in enumerate(itertools.product(*self.actions))}
            self.action_space = gym.spaces.Discrete(len(self.action_map))

        # get initial state
        self.reset()
        
    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, done, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            done (bool) :
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
            'charge_rates': [],
            'elec_cost': None,
            'best_possible_energy': [],
            'tot_energy_delivered': [],
            'price': None,
            'energy_delivered': 0
        }
        if not self.config.continuous_actions:
            #translate action from number to tuple
            action = self.action_map[action]
        if self.config.scale_actions_transformer:
            action = utils.scale_action(action, self.transformer_capacity)
        new_state, reward = self.take_action(action)
        #translate action from number to tuple
        self.info['new_state'] = new_state

        return self.featurize(new_state), reward, self.done, self.info
    
    def charge_car(self, station, new_station, charge_rate):
        """
        Charges the car that was present at station
        :param station: (dictionary) contains old station attributes
        :param new_station: (dictionary) contains to be updated station attributes
        :param charge_rate: (float)
        :return: (float) amount of energy charged to the car
        """
        is_car, des_char, per_char, curr_dur =  station['is_car'], station['des_char'], station['per_char'], station['curr_dur']
        new_station['is_car'] = True
        new_station['des_char'] = des_char
        new_station['curr_dur'] = curr_dur + self.time_step
        curr_char = per_char*des_char
        total_char = max(min(des_char, curr_char +charge_rate*self.time_step), 0)
        energy_added = total_char - curr_char
        self.info['energy_delivered'] += energy_added
        if energy_added > 0:  # why is this if / elif / else necessary?
            self.info['charge_rates'].append(charge_rate)
        elif energy_added < 0:
            self.info['charge_rates'].append(charge_rate)
        else:
            self.info['charge_rates'].append(0)
        new_station['per_char'] = float(total_char)/des_char
        return energy_added
    
    def car_leaves(self, new_station):
        """
        Handles a car leaving at the current time step
        :param new_station: (dictionary) contains to be updated station attributes
        :return:
        """
        #compute statistics for self.info
        total_energy = new_station['des_char']*new_station['per_char']
        best_possible_energy = min(new_station['curr_dur']*self.max_power, new_station['des_char'])
        self.info['tot_energy_delivered'].append(total_energy)
        self.info['best_possible_energy'].append(best_possible_energy)
        #reset the station
        new_station['is_car'] = False
        new_station['des_char'], new_station['per_char'], new_station['curr_dur'] = 0,0,0
        if self.config.end_after_leave and self.config.NUM_STATIONS == 1:
            self.done = True

    def car_arrives(self, new_station, session):
        """
        Handles a new car arriving at the current time step
        :param new_station: (dictionary) contains to be updated station attributes
        :param session: (list) charging data for new charging session
        :return:
        """
        new_station['is_car'] = True
        new_station['des_char'] = session[1]
        new_station['per_char'] = 0
        new_station['curr_dur'] = 0
    
    def take_action(self, actions):
        """
        Charges the cars at each station as specified by the charging rates in actions
        :param actions: (list) charge rates at each station as specified by the controller
        :return: new state (dict) and reward (float) of taking the charging actions
        """
        # print(self.state)
        new_state = {}
        time = self.state['time']
        new_time = time + datetime.timedelta(hours=self.time_step)
        new_state['time'] = new_time
        new_state["price"] = self.elec_price_data[new_time.to_pydatetime()]
        stations, energy_charged, percent_charged = [],[],[]
        for stn_num, station in enumerate(self.state['stations']):
            new_station = deepcopy(station)
            if station['is_car']:
                energy_added = self.charge_car(station, new_station, actions[stn_num])
                if new_station['curr_dur'] >= self.durations[stn_num]:
                    self.durations[stn_num] = 0
                    self.car_leaves(new_station)
            else:
                energy_added = 0
                self.info['charge_rates'].append(0)
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
            percent_charged=percent_charged,
        )
        # if self.config.penalize_unecessary_actions > 0:
            # is_car = np.array([int(station['is_car']) for station in self.state['stations']])
            # not_full = np.array([int(p < 1.0) for p in percent_charged])
            # a_charge = np.array([int(a > 0.1) for a in actions])
            # unnecessary_actions = (1 - (is_car*not_full)) * a_charge
            # # unnecessary_actions = (1 - is_car) * a_charge  # not penalize charging when full car is present
            # reward -= self.config.penalize_unecessary_actions * float(sum(unnecessary_actions)) / float(self.num_stations)
        self.state = new_state
        if (not self.config.end_after_leave) or (self.config.NUM_STATIONS > 1):
            self.done = sum([len(loc) for loc in self.charging_data]) + sum(self.durations) == 0
        return new_state, reward

    def get_initial_state(self):
        """
        Sets the initial state based on the earliest charging session in self.charging_data
        :return:
        """
        ## get_start_time
        initial_state = {}
        start_time = min([loc[-1][0] for loc in self.charging_data if len(loc) > 0])
        initial_state["time"] = start_time
        initial_state["price"] = self.elec_price_data[start_time.to_pydatetime()]
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

    def reward(self, energy_charged, percent_charged):
        """
        Calculates the reward based on the energy charged at each station during the time step and the percent charged
        :param energy_charged: (list) energy charged at each station during the time step
        :param percent_charged: (list) percent charged for each car at each station
        :return: the reward for the current time step
        """
        # charging_powers = print(self.info['charge_rates'])
        magnitude = self.config.reward_magnitude
        if self.config.charge_empty_factor > 0:
            charge_influence = 1.0 + self.config.charge_empty_factor * (0.5 - np.array(percent_charged))
        else:
            charge_influence = 1

        if self.config.use_delayed_charge_reward == "leave":
            charge_reward = 0
            for i in range(self.num_stations):
                self.delayed_charge_reward[i] += energy_charged[i]
                if self.done or not self.get_current_state()['stations'][i]['is_car']:
                    charge_reward += self.delayed_charge_reward[i]
                    self.delayed_charge_reward[i] = 0
        elif self.config.use_delayed_charge_reward == "full":
            charge_reward = 0
            for i in range(self.num_stations):
                self.delayed_charge_reward[i] += energy_charged[i]
                if percent_charged[i] >= 1.0 and energy_charged[i] > 0:
                    charge_reward += self.delayed_charge_reward[i]
                    self.delayed_charge_reward[i] = 0
                if not self.get_current_state()['stations'][i]['is_car']:
                    self.delayed_charge_reward[i] = 0
        elif self.config.use_delayed_charge_reward == "full_bonus":
            charge_reward = np.sum(energy_charged * charge_influence)
            for i in range(self.num_stations):
                if percent_charged[i] >= 1.0 and energy_charged[i] > 0:
                    charge_reward += 5 * self.time_step * self.max_power * charge_influence
        else:
            charge_reward = np.sum(energy_charged * charge_influence)


        elec_price = self.elec_price_data[self.get_current_state()['time'].to_pydatetime()]
        self.info['price'] = elec_price
        elec_cost = np.sum(energy_charged) * elec_price
        #store statistics
        self.info['elec_cost'] = elec_cost

        pow_penalty = 0.0
        if not self.config.scale_actions_transformer:
            capa = self.transformer_capacity
            if self.config.solar_behind_meter > 0:
                # with elec_price as inverse of solar output, increase transformer capa relative to solar generation
                capa = capa * (1 + self.config.solar_behind_meter * (1 - elec_price))
            pow_violation = (np.sum(energy_charged) / self.time_step) - capa
            if pow_violation > 0:
                pow_ratio = min(1, pow_violation / capa)
                pow_penalty = np.exp(np.log(magnitude)*pow_ratio) - 1.0  # [0, 1000]

        divider = self.time_step*self.transformer_capacity
        charge_reward = charge_reward / divider  # [0, 1]
        elec_cost = elec_cost / divider  # [0, 1] if price [0,1]

        reward = [magnitude*charge_reward, -1*magnitude*elec_cost, -1*pow_penalty]
        # print(reward)
        reward = sum([r*w for r, w in zip(reward, self.reward_weights)])
        return reward

    def get_current_state(self):
        """
        Returns the current state
        :return: state (dict) the current state
        """
        return self.state

    def sample_data(self):
        """
        Samples a given time period (based on config.EVAL_EPS_LEN) of data to simulate over.
        Samples from evaluation data depending on self.evaluation_mode
        :return: returns the charging data and electric price data over the given time period
        """
        elec_price_data = toy_data.price

        if self.evaluation_mode:
            charging_data = utils.sample_charging_data(
                self.eval_charging_data,
                self.config.EVAL_EPS_LEN,
                self.time_step,
                self.random_state
            )
        else:
            charging_data = utils.sample_charging_data(
                self.train_charging_data,
                self.episode_length,
                self.time_step,
                self.random_state
            )
        return charging_data, elec_price_data

    def reset(self):
        """
        Resets the environment
        :return:
        """
        self.done = False
        self.info = None
        self.durations = []
        self.charging_data, self.elec_price_data = self.sample_data()
        self.state = self.get_initial_state()
        self.delayed_charge_reward = [0] * self.num_stations
        featurized_state = self.featurize(self.state)
        return featurized_state

    def render(self, mode='human', close=False):
        """Used for vizualizing charging, not implemented"""
        pass

    def close(self):
        """Not implemented"""
        pass

    def seed(self, seed=None):
        """Not implemente"""
        # return self.seed(seed)
        pass
