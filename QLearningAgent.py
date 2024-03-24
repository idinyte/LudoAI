import numpy as np

import numpy as np
from qTable import Rewards
from stateSpace import Action, State, StateSpace


class QLearningAgent(StateSpace):

    def __init__(self, player_i, gamma, learning_rate):
        super().__init__()
        self.q_learning = Rewards(len(State), len(Action), gamma=gamma, lr=learning_rate)
        self.player_i = player_i

    def update(self, players, pieces_to_move, dice):
        super().update(players, self.player_i, pieces_to_move, dice)
        state, action = self.q_learning.get_next_action(self.action_table_obj.action_table)
        pieces_to_move = self.action_table_obj.get_piece_to_move(state, action)
        self.state = state
        self.action = action

        return pieces_to_move

    def reward(self, players, pieces_to_move):
        super().get_possible_actions(players, self.player_i, pieces_to_move)
        new_action_table = np.nan_to_num(self.action_table_obj.action_table, nan=0.0)
        self.q_learning.reward(self.state, new_action_table, self.action)