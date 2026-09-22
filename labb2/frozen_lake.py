import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TransformReward
import qlearning

# https://gymnasium.farama.org/environments/toy_text/frozen_lake/
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode=None)
env = TransformReward(env, lambda r: 1 if r == 1 else r-0.04)
q_table = qlearning.QTable(_gamma=0.99, _alpha=0.5, _epsilon=1.0, _action_size=env.action_space.n, _state_size=env.observation_space.n)

observation, info = env.reset(seed=42)
action = env.action_space.sample()
print("Starting")
terminated_i = 0

### TRAINING LOOP
# In this loop your agent will be trained without showing the run
# To stop this training loop you can pres ctrl+c in the terminal. This will start the testing loop below
# You can implement a stop criteria if you want
try:
    while True:
        observation, reward, terminated, truncated, info = env.step(action)
        action = q_table.update_Q(observation, reward)  ### <-- This line samples a random action for from the environment. Replace this with your Qlearning algorithm check ###

        if terminated or truncated:         ### <-- This condition triggers if the run is terminated, as it the agent reaches a goal or falls into a hole
            terminated_i = terminated_i + 1


            if reward > 0:                  ### <-- This condition triggers when the agent finds the goal. You are free to debug and print how you want
                print("Found goal at iteration", terminated_i)

            observation, info = env.reset()
            q_table.previous_Q_index = np.nan  # Reset previous state to null
            action = q_table.epsilon_greedy(observation)  # Sample a random action to start the next run
except KeyboardInterrupt:
    print("Run ended")
    env.close()


### TESTING LOOP
# Put your algorithm for taking the best action here
# To stop this testing loop, press ctrl+c in the terminal.
env_test = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode='human')
observation, info = env_test.reset(seed=42)
try:
    while True:
        observation, reward, terminated, truncated, info = env_test.step(action)
        action = q_table.get_best_action(observation)  ### <-- This line samples a random action for from the environment. Replace this with your optimal action calculation ###

        if terminated or truncated:
            observation, info = env_test.reset()

except KeyboardInterrupt:
    print("Test ended")
    env_test.close()