import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0")
n_states = 40

def observation_to_state(observation):
    # observation - (x pos of the car, velocity)
    print(env.observation_space)

observation_to_state(1)
q_table = np.zeros((n_states, n_states, env.action_space.n))

