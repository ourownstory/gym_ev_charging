from gym.envs.registration import register

register(
    id='ev-charging-v0',
    entry_point='gym_ev_charging.envs:EVChargingEnv',
)

register(
    id='ev-charging-v1',
    entry_point='gym_ev_charging.envs:EVChargingEnvVec',
)
