import numpy as np

class ActionTable():
    action_table, pieces_table = None, None

    def __init__(self, states, actions, Action):
        self.states = states
        self.actions = actions
        self.Action_obj = Action
        self.reset()

    def get_piece_to_move(self, state, action, piece_pos):
        pieces = self.pieces_table[state, action]
        if len(pieces) == 1:
            return pieces[0]
        
        available_options = []
        for i, pos in enumerate(piece_pos):
            if i in pieces:
                available_options.append((i, pos))
        sorted_options = sorted(available_options, key=lambda x: x[1])
        
        
        if action != self.Action_obj.SAFE_Die.value or action != self.Action_obj.UNSAFE_Die.value:
            piece_to_move = sorted_options[-1][0] # move the farthest piece if there are multiple pieces that can do the same action
        else:
            piece_to_move = sorted_options[0][0] # move closest piece if it is going to die
            
        # print(f"action {action} sorted_options {sorted_options} pieces {pieces} move {piece_to_move}")
        return piece_to_move
         

    def reset(self):
        self.action_table = np.full((self.states, self.actions), np.nan)
        self.pieces_table = np.full((self.states, self.actions), None, dtype=object)
        for i in range(self.states):
            for j in range(self.actions):
                self.pieces_table[i, j] = []

    def update_action_table(self, state, action, piece_i):
        self.action_table[state.value, action.value] = 1
        self.pieces_table[state.value, action.value].append(piece_i)