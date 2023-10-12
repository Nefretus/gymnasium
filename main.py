import gymnasium as gym
import numpy as np
from tqdm import tqdm
from collections import defaultdict

env = gym.make("MountainCar-v0")
n_states = 40
n_iter = 10000

initial_eps = 1.0
eps = initial_eps
eps_decay = initial_eps / (n_iter / 2)

min_eps = 0.1
gamma = 1.0
lr = 0.1

def observation_to_state(observation):
    env_step = (env.observation_space.high - env.observation_space.low) / n_states
    idx_x = int((observation[0] - env.observation_space.low[0]) / env_step[0])
    idx_y = int((observation[1] - env.observation_space.low[1]) / env_step[1])
    return (idx_x, idx_y)

def create_policy(q_table):
    policy = defaultdict(int)
    for state, q_values in q_table.items():
        policy[state] = np.argmax(q_values)
    return policy


q_table = defaultdict(lambda: np.zeros(env.action_space.n))
action = 0

for episode in tqdm(range(n_iter)):
    observation, _ = env.reset()
    eps = max(min_eps, eps - eps_decay)
    for _ in range(n_iter):
        state = observation_to_state(observation)
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))
        observation, reward, terminated, truncated, _ = env.step(action)
        next_state = observation_to_state(observation)
        q_table[state][action] = q_table[state][action] + lr * (
                    reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
        if terminated: break
env.close()

# visualization
env = gym.make("MountainCar-v0", render_mode='human')
policy = create_policy(q_table)
observation, _ = env.reset()
for _ in range(n_iter):
    env.render()
    state = observation_to_state(observation)
    action = policy[state]
    observation, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated: break
