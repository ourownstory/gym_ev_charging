import gym
import gym_ev_charging
from gym_ev_charging.config_gym import get_config 
env_config = get_config('discrete')
env = gym.make('ev-charging-v0')
# env = gym.make('CartPole-v0')
env.build(env_config)
for i_episode in range(3):
    observation = env.reset()
    for t in range(10):
#        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print (energy_charged, percent_charged)
        print (observation, '\n', reward)
        for stn in info['new_state']['stations']:
            print (stn)
        print (info['charge_rates'])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    print ("reset")
