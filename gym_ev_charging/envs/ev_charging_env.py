import gym
from gym import error, spaces, utils
# from gym.utils import seeding
import numpy as np

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
        self.episode_over = False
        self.action_space = gym.spaces.Discrete(2)
        self.state = np.zeros(10)
        self.observation_space = self.state
        self.reward_range = (0, 1000)
        self.total_steps = 0
        self.max_state = 10

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
        self.total_steps += 1

        if action == 0:
            self.state[0] = max(0, self.state[0] - 1)
        elif action == 1:
            self.state[0] = min(self.max_state, self.state[0] + 1)
        else:
            raise("action not defined")

        reward = 1000*int(self.state[0] == self.max_state)
        reward += -np.random.random()
        # reward *= (1.0 / self.total_steps)

        episode_over = self.state[0] == self.max_state

        ob = self.state + np.random.random()
        return ob, reward, episode_over, {}

    def reset(self):
        self.state = np.zeros(10)
        return self.state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        # return self.seed(seed)
        pass
