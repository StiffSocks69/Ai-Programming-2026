import numpy as np

np.seterr(all='raise')

### This file contains suggested skeleton code if you do not know how to start ###
##  You do not have to follow this skeleton code, you can use your own structure ##


class QTable:
    def __init__(self, _gamma, _alpha, _epsilon, _action_size, _state_size):

        self.alpha = _alpha
        self.gamma = _gamma
        self.epsilon = _epsilon
        self.action_size = _action_size
        self.initial_array_size = _state_size


        # s, a, r , the previous state, action, and reward, initially null
        self.previous_Q_index = np.nan
        self.previous_action = np.nan
        self.previous_reward = np.nan

        # Initiate Q-tables here
        self.Q = np.zeros((self.initial_array_size, self.action_size))  # Create a Q-table with the size of the state space and action space


    def get_Q_index(self, coded_state):
        return coded_state  # Return the index of a row in the Qtable

    def get_Q_actions(self, Q_index):
        return self.Q[Q_index]  # Return action for a row in the Qtable

    def set_Q(self, Q_index, action, new_Q):
        self.Q[Q_index, action] = new_Q  # Set calculated Q value for a state action pair

    def get_max_Q(self, Q_index):
        return np.max(self.Q[Q_index])  # Maxumum Q value in a row

    def epsilon_greedy(self, Q_index):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)  # Random action
        return np.argmax(self.Q[Q_index])  # Best action

    def get_best_action(self, current_state):

        return np.argmax(self.Q[current_state])  # Best action

    def update_Q(self, current_state, current_reward):
        # Algorithm used from "Artificial Intelligence A Modern Approach" by Stuart Russell and Peter Norvig
        # Q-Learning-Agent, page 844, figure 21.8
        if not np.isnan(self.previous_Q_index):
            # Calculate new Q value
            new_Q = self.Q[self.previous_Q_index, self.previous_action] + self.alpha * (
                    current_reward + self.gamma * self.get_max_Q(current_state) - self.Q[
                self.previous_Q_index, self.previous_action])

            # Update Q-table with new Q value
            self.set_Q(self.previous_Q_index, self.previous_action, new_Q)
        self.previous_Q_index = current_state  # Update previous state to current state
        next_action = self.epsilon_greedy(current_state)  # Action to take
        self.previous_action = next_action  # Action to take
        self.previous_reward = current_reward  # Update previous reward to current reward
        return next_action  # Action to take for next step AFTER epsilon greedy has been used
