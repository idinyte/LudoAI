import numpy as np

import numpy as np
from Qlearning.qTable import Rewards
from Qlearning.stateSpace import Action, State, StateSpace


class QLearningAgent(StateSpace):
    state, action = -1, -1
    def __init__(self, player_i, gamma, learning_rate, epsilon):
        super().__init__()
        self.q_learning = Rewards(len(State), len(Action), epsilon, gamma, learning_rate)
        self.player_i = player_i

    def update(self, dice, pieces_to_move, players):
        super().update_state_space(players, self.player_i, pieces_to_move, dice)
        state, action = self.q_learning.get_next_action(self.action_table_obj.action_table)
        piece_to_move = self.action_table_obj.get_piece_to_move(state, action, players[self.player_i].pieces)
        self.state = state # int
        self.action = action # int

        return piece_to_move

    def reward(self, players, pieces_to_move):
        super().get_possible_actions(players, self.player_i, pieces_to_move)
        future_action_table = np.nan_to_num(self.action_table_obj.action_table, nan=0.0)
        self.q_learning.reward(self.state, self.action, future_action_table)
        
    def print_state_action(self):
        print(f"Number of unsafe pieces = {self.state}")
        action_str = ""
        match self.action:
            case Action.SAFE_MoveOut.value:
                action_str = "SAFE_MoveOut"
            case Action.SAFE_KillMoveOut.value:
                action_str = "SAFE_KillMoveOut"
            case Action.SAFE_Dice.value:
                action_str = "SAFE_Dice"
            case Action.SAFE_Star.value:
                action_str = "SAFE_Star"
            case Action.SAFE_Globe.value:
                action_str = "SAFE_Globe"
            case Action.SAFE_DoubleUp.value:
                action_str = "SAFE_DoubleUp"    
            case Action.SAFE_Kill.value:
                action_str = "SAFE_Kill"    
            case Action.SAFE_KillStar.value:
                action_str = "SAFE_KillStar"    
            case Action.SAFE_Die.value:
                action_str = "SAFE_Die"    
            case Action.SAFE_GoalZone.value:
                action_str = "SAFE_GoalZone"    
            case Action.SAFE_Finish.value:
                action_str = "SAFE_Finish"    
            case Action.UNSAFE_Dice.value:
                action_str = "UNSAFE_Dice"    
            case Action.UNSAFE_Star.value:
                action_str = "UNSAFE_Star"    
            case Action.UNSAFE_Globe.value:
                action_str = "UNSAFE_Globe"
            case Action.UNSAFE_DoubleUp.value:
                action_str = "UNSAFE_DoubleUp"   
            case Action.UNSAFE_Kill.value:
                action_str = "UNSAFE_Kill"   
            case Action.UNSAFE_KillStar.value:
                action_str = "UNSAFE_KillStar"   
            case Action.UNSAFE_Die.value:
                action_str = "UNSAFE_Die"   
            case Action.UNSAFE_GoalZone.value:
                action_str = "UNSAFE_GoalZone"   
            case Action.UNSAFE_Finish.value:
                action_str = "UNSAFE_Finish"
        print(f"Action: {action_str}")
        return action_str