import random
import numpy as np
from stateSpace import State, Action

class Rewards():
    rewards_table, q_table = None, None
    max_expected_reward = 0
    def __init__(self, states, actions, epsilon, gamma, lr):
        self.q_table = np.zeros([states, actions])
        self.rewards_table = np.zeros([states, actions])
        
        self.epsilon_greedy = epsilon
        self.gamma = gamma
        self.lr = lr

        self.rewards_table[State.SAFE][Action.MoveOut.value] = 0.1
        self.rewards_table[State.SAFE][Action.Dice.value] = -0.4
        self.rewards_table[State.SAFE][Action.Star.value] = 0.4
        self.rewards_table[State.SAFE][Action.Globe.value] = 0.4
        self.rewards_table[State.SAFE][Action.DoubleUp.value] = 0.4
        self.rewards_table[State.SAFE][Action.Kill.value] = 1.2
        self.rewards_table[State.SAFE][Action.KillStar.value] = self.rewards_table[State.SAFE][Action.Star.value] + self.rewards_table[State.SAFE][Action.Kill.value]
        self.rewards_table[State.SAFE][Action.KillMoveOut.value] = self.rewards_table[State.SAFE][Action.Kill.value] + self.rewards_table[State.SAFE][Action.MoveOut.value]
        self.rewards_table[State.SAFE][Action.Die.value] = -1.6
        self.rewards_table[State.SAFE][Action.GoalZone.value] = 0
        self.rewards_table[State.SAFE][Action.Finish.value] = 0.2

        self.rewards_table[State.UNSAFE][Action.MoveOut.value] = 0
        self.rewards_table[State.UNSAFE][Action.Dice.value] = 0.3
        self.rewards_table[State.UNSAFE][Action.Star.value] = 0.7
        self.rewards_table[State.UNSAFE][Action.Globe.value] = 0.8
        self.rewards_table[State.UNSAFE][Action.DoubleUp.value] = 0.8
        self.rewards_table[State.UNSAFE][Action.Kill.value] = 1.2
        self.rewards_table[State.UNSAFE][Action.KillStar.value] = self.rewards_table[State.UNSAFE][Action.Star.value] + self.rewards_table[State.UNSAFE][Action.Kill.value]
        self.rewards_table[State.UNSAFE][Action.KillMoveOut.value] = self.rewards_table[State.UNSAFE][Action.Kill.value] + self.rewards_table[State.UNSAFE][Action.MoveOut.value]
        self.rewards_table[State.UNSAFE][Action.Die.value] = -1.2
        self.rewards_table[State.UNSAFE][Action.GoalZone.value] = 1
        self.rewards_table[State.UNSAFE][Action.Finish.value] = 2

    def get_next_action(self, action_table):
        q_table_options = np.multiply(self.q_table, action_table.flatten())
    
        if random.uniform(0, 1) < self.epsilon_greedy:
            non_nan_state_actions = np.argwhere(~np.isnan(action_table))
            state, action = non_nan_state_actions[np.random.randint(0, len(non_nan_state_actions))]
        else:
            state, action = np.unravel_index(np.nanargmax(q_table_options), q_table_options.shape)

        return (state, action)


    def reward(self, state, new_action_table, action):
        reward = self.rewards_table[state, action]
        estimate_of_optimal_future_value = np.max(self.q_table * new_action_table)
        old_q_value = self.q_table[state, action]
        delta_q = self.lr * (reward + self.gamma * estimate_of_optimal_future_value - old_q_value)
        self.max_expected_reward += reward
        self.q_table[state, action] = old_q_value + delta_q
    