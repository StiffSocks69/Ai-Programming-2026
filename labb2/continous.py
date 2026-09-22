import gymnasium as gym
#import qlearning 

# https://gymnasium.farama.org/environments/classic_control/acrobot/
env = gym.make('Acrobot-v1', render_mode=None)

# Or
# https://gymnasium.farama.org/environments/box2d/lunar_lander/
#env = gym.make("LunarLander-v2", render_mode=None)


observation, info = env.reset()
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
        action = env.action_space.sample()  ### <-- This line samples a random action for from the environment. Replace this with your Qlearning algorithm ###

        if terminated or truncated:         ### <-- This condition triggers if the run is terminated, as it the agent reaches a goal or falls into a hole
            terminated_i = terminated_i + 1

            if reward == 0:                  ### <-- This condition triggers when the agent finds the goal. You are free to debug and print how you want
                print("Found goal at iteration", terminated_i)

            observation, info = env.reset()
except KeyboardInterrupt:
    print("Run ended")
    env.close()


### TESTING LOOP
# Put your algorithm for taking the best action here
# To stop this testing loop, press ctrl+c in the terminal.
env_test = gym.make('Acrobot-v1', render_mode='human')

#env_test = gym.make("LunarLander-v2", render_mode='human')

observation, info = env_test.reset()
try:
    while True:
        observation, reward, terminated, truncated, info = env_test.step(action)
        action = env_test.action_space.sample()  ### <-- This line samples a random action for from the environment. Replace this with your optimal action calculation ###

        if terminated or truncated:
            observation, info = env_test.reset()

except KeyboardInterrupt:
    print("Test ended")
    env_test.close()