from gym.envs.registration import register

register(
    id='evcharging-v0',
    entry_point='gym_evcharging.envs:EVDummyEnv',
)
