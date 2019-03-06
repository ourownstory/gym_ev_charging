from gym.envs.registration import register

register(
    id='dummy-v0',
    entry_point='gym_ev_charging.envs:DummyEnv',
)

register(
    id='ev-charging-v0',
    entry_point='gym_ev_charging.envs:EVChargingEnv',
)
