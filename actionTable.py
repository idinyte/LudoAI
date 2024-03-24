import numpy as np

class ActionTable():
    action_table, pieces_table = None, None

    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.reset()

    def get_piece_to_move(self, state, action):
        # if state < 0 or action < 0:
        #     return -1

        return self.pieces_table[state, action]

    def reset(self):
        self.action_table = np.full((self.states, self.actions), np.nan)
        self.pieces_table= np.full((self.states, self.actions), np.nan)

    def update_action_table(self, state, action, piece_i):
        self.action_table[state.value, action.value] = 1
        self.pieces_table[state.value, action.value] = piece_i