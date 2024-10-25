import numpy as np

class ActionTable():
    action_table = None

    def __init__(self, actions):
        self.actions = actions
        self.reset()       

    def reset(self):
        self.action_table = np.full((self.actions), -1)

    def update_action_table(self, action, piece_i):
        self.action_table[piece_i] = int(action.value)