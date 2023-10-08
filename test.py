from collections import defaultdict
import numpy as np
from tqdm import tqdm
import gymnasium as gym

env = gym.make('Blackjack-v1', render_mode='human')

class BlackjackAgent:
    def __init__(
            self,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95
    ):
        # initlize a RL agent with empty dictionary of state-action value, a learing rate and epsilon
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, observation: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[observation]))
        # we need to lower this propability, so that agent is more confident on decisions made, insted of exploring other options

    def update(self, observation, action, reward, terminated, next_observation):
        # update the Q-value of an actions
        future_q_value = (not terminated) * np.max(self.q_values[next_observation]) #maximum future award
        temp_difference = (
                reward + self.discount_factor * future_q_value - self.q_values[observation][action]
        )
        self.q_values[observation][action] = (
                self.q_values[observation][action] + self.lr * temp_difference
        )
        # best possible reward after performing an action at the state "s"

    def decay_epsilon(self):
        self.epislon = max(self.final_epsilon, self.epsilon_decay)

learning_rate = 0.01
n_episodes = 100_000
initial_epsilon = 1.0
epsilon_decay = initial_epsilon / (n_episodes / 2)
final_epsilon = 0.1

agent = BlackjackAgent(
    learning_rate=learning_rate,
    initial_epsilon=initial_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon
)

def obs_to_state(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / 40
    a = int((obs[0] - env_low[0]) / env_dx[0])
    b = int((obs[1] - env_low[1]) / env_dx[1])
    print(env_low, env_high, env_dx, a, b)
    return a, b

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
print(env.observation_space)

# for episode in tqdm(range(n_episodes)):
#     print('???????????')
#     observation, info = env.reset()
#
#     done = False
#     while True:
#         action = agent.get_action(observation)
#         next_observation, reward, terminated, truncated, info = env.step(action)
#         agent.update(observation, action, reward, terminated, next_observation)
#         env.render()
#         if terminated or truncated: break;
#         observation = next_observation
#
#     agent.decay_epsilon()

