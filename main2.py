from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm

env = gym.make('CartPole-v1')

learning_rate = 0.1
discount = 0.95
timestamps = 10000
n_states = 40
q_table = defaultdict(lambda : np.zeros(env.action_space.n))
terminated = False
min_eps = 0.1
gamma = 1.0

initial_eps = 1.0
eps = initial_eps
eps_decay = initial_eps / (timestamps / 2)


def discretizer(observation):
    discrete_angle = int(observation[2] * 10000 / 41.8)
    discrete_velocity = int(observation[3] * 50)
    return discrete_angle, discrete_velocity


def policy(state: tuple):
    return np.argmax(q_table[state])
def create_policy(q_table):
    policy = defaultdict(int)
    for state, q_values in q_table.items():
        policy[state] = np.argmax(q_values)
    return policy

def new_q_value(reward: float, new_state: tuple, discount_factor=1) -> float:
    future_optimal_value = np.max(q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value

for episode in tqdm(range(timestamps)):
    observation, _ = env.reset()
    eps = max(min_eps, eps - eps_decay)
    for _ in range(timestamps):
        state = discretizer(observation)
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))
        observation, reward, terminated, truncated, _ = env.step(action)
        next_state = discretizer(observation)
        q_table[state][action] = q_table[state][action] + learning_rate * (
                    reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
        if terminated: break
env.close()

# visualization
env = gym.make('CartPole-v1', render_mode="human")
policy = create_policy(q_table)
observation, _ = env.reset()
for _ in range(timestamps):
    env.render()
    state = discretizer(observation)
    action = policy[state]
    observation, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated: break
