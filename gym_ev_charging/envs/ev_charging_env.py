import numpy as np
from copy import deepcopy
from toy_data import *
from utils import *
import config
import gym
from gym import error, spaces, utils
# from gym.utils import seeding
import numpy as np
from copy import deepcopy


class EVChargingEnv(gym.Env):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    When implementing an environment, override the following methods
    in your subclass:

        _step
        _reset
        _render
        _close
        _seed

    And set the following attributes:
a
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.num_stations = config.NUM_STATIONS
        self.episode_length = config.EPS_LEN
        self.time_step = config.TIME_STEP
        self.transformer_capacity = config.TRANSFORMER_CAPACITY
        self.reward_weights = config.REWARD_WEIGHTS
        self.total_charging_data = config.TOTAL_CHARGING_DATA
        self.total_elec_price_data=pd.DataFrame()
        self.random_state = np.random.RandomState(config.RAND_SEED)
        
        #gym stuff 
        self.episode_over = False
        self.action_space = gym.spaces.Box(low = 0, high = 1, shape = config.NUM_STATIONS, dtype = 0)
        self.observation_space = self.state
        self.total_steps = 0

        self.start_time = None
        self.charging_data = None
        self.elec_price_data = None
        self.durations = []
        self.done = False
        self.state = None
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
        ob, reward = self.take_action(action) 
        episode_over = self.done
        return ob, reward, episode_over, {}
    
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

    def sample_data(self):
        # self.charging_data = deepcopy(locations)
        self.elec_price_data = price

        valid_dates = self.total_charging_data.loc[self.total_charging_data.index[0]:self.total_charging_data.index[-1] - datetime.timedelta(hours=self.episode_length * self.time_step)].index
        self.start_time = valid_dates[self.random_state.choice(range(len(valid_dates)))].to_pydatetime()

        self.charging_data = sample_charging_data(self.total_charging_data, self.start_time, self.episode_length, self.time_step)

    def reset(self):
        self.done = False
        self.durations = []
        self.sample_data()
        self.state = self.get_initial_state()
        return self.state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        # return self.seed(seed)
        pass
