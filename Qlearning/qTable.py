import random
import numpy as np
from Qlearning.stateSpace import State, Action

class Rewards():
    rewards_table, q_table = None, None
    def __init__(self, states, actions, epsilon, gamma, lr):
        self.rewards_table = np.zeros([actions])
        self.epsilon_greedy = epsilon
        self.gamma = gamma
        self.lr = lr

        self.rewards_table[Action.SAFE_MoveOut.value] = 0.5
        self.rewards_table[Action.SAFE_Dice.value] = -0.4
        self.rewards_table[Action.SAFE_Star.value] = 0.4
        self.rewards_table[Action.SAFE_Globe.value] = 0.4
        self.rewards_table[Action.SAFE_DoubleUp.value] = 0.4
        self.rewards_table[Action.SAFE_Kill.value] = 1.2
        self.rewards_table[Action.SAFE_KillStar.value] = 1.6
        self.rewards_table[Action.SAFE_KillMoveOut.value] = 1.7
        self.rewards_table[Action.SAFE_Die.value] = -1.6
        self.rewards_table[Action.SAFE_GoalZone.value] = -0.8
        self.rewards_table[Action.SAFE_Finish.value] = 0.4

        self.rewards_table[Action.UNSAFE_Dice.value] = 0
        self.rewards_table[Action.UNSAFE_Star.value] = 0.8
        self.rewards_table[Action.UNSAFE_Globe.value] = 0.8
        self.rewards_table[Action.UNSAFE_DoubleUp.value] = 0.8
        self.rewards_table[Action.UNSAFE_Kill.value] = 1.2
        self.rewards_table[Action.UNSAFE_KillStar.value] = 2
        self.rewards_table[Action.UNSAFE_Die.value] = -1.2
        self.rewards_table[Action.UNSAFE_GoalZone.value] = 1
        self.rewards_table[Action.UNSAFE_Finish.value] = 2
        self.q_table = np.array([self.rewards_table.copy()*0.01 for _ in range(states)])

    def get_next_action(self, action_table):
        q_table_options = np.multiply(self.q_table, action_table)
    
        if random.uniform(0, 1) < self.epsilon_greedy:
            non_nan_state_actions = np.argwhere(~np.isnan(action_table))
            state, action = non_nan_state_actions[np.random.randint(0, len(non_nan_state_actions))]
        else:
            state, action = np.unravel_index(np.nanargmax(q_table_options), q_table_options.shape)

        return state, action
    
    def update_epsilon(self, epsilon, decay_rate, episode):
        self.epsilon_greedy = epsilon * np.exp(-decay_rate*episode)

    def reward(self, state, action, future_action_table):
        reward = self.rewards_table[action]
        estimate_of_optimal_future_value = np.max(self.q_table * future_action_table)
        old_q_value = self.q_table[state, action]
        delta_q = self.lr * (reward + self.gamma * estimate_of_optimal_future_value - old_q_value)
        self.q_table[state, action] = old_q_value + delta_q
    