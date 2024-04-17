import numpy as np
from Qlearning.stateSpace import Action

class Rewards():
    rewards_table = None
    def __init__(self, actions):
        self.rewards_table = np.zeros([actions])

        self.rewards_table[Action.SAFE_MoveOut.value] = 1
        self.rewards_table[Action.SAFE_Dice.value] = -0.1
        self.rewards_table[Action.SAFE_Star.value] = 0.8
        self.rewards_table[Action.SAFE_Globe.value] = 0.4
        self.rewards_table[Action.SAFE_DoubleUp.value] = 0.4
        self.rewards_table[Action.SAFE_Kill.value] = 2.4
        self.rewards_table[Action.SAFE_KillStar.value] = 3.2
        self.rewards_table[Action.SAFE_KillMoveOut.value] = 4
        self.rewards_table[Action.SAFE_Die.value] = -1.6
        self.rewards_table[Action.SAFE_GoalZone.value] = -0.8
        self.rewards_table[Action.SAFE_Finish.value] = 1

        self.rewards_table[Action.UNSAFE_Dice.value] = -0.05
        self.rewards_table[Action.UNSAFE_Star.value] = 0.8
        self.rewards_table[Action.UNSAFE_Globe.value] = 0.8
        self.rewards_table[Action.UNSAFE_DoubleUp.value] = 0.8
        self.rewards_table[Action.UNSAFE_Kill.value] = 3.2
        self.rewards_table[Action.UNSAFE_KillStar.value] = 4
        self.rewards_table[Action.UNSAFE_Die.value] = -1.2
        self.rewards_table[Action.UNSAFE_GoalZone.value] = 1
        self.rewards_table[Action.UNSAFE_Finish.value] = 2
        
        self.rewards_table *= 1000

# self.rewards_table[Action.SAFE_MoveOut.value] = 1
# self.rewards_table[Action.SAFE_Dice.value] = -0.4
# self.rewards_table[Action.SAFE_Star.value] = 0.4
# self.rewards_table[Action.SAFE_Globe.value] = 0.4
# self.rewards_table[Action.SAFE_DoubleUp.value] = 0.4
# self.rewards_table[Action.SAFE_Kill.value] = 1.2
# self.rewards_table[Action.SAFE_KillStar.value] = 1.6
# self.rewards_table[Action.SAFE_KillMoveOut.value] = 1.7
# self.rewards_table[Action.SAFE_Die.value] = -1.6
# self.rewards_table[Action.SAFE_GoalZone.value] = -0.8
# self.rewards_table[Action.SAFE_Finish.value] = 0.4

# self.rewards_table[Action.UNSAFE_Dice.value] = 0.1
# self.rewards_table[Action.UNSAFE_Star.value] = 0.8
# self.rewards_table[Action.UNSAFE_Globe.value] = 0.8
# self.rewards_table[Action.UNSAFE_DoubleUp.value] = 0.8
# self.rewards_table[Action.UNSAFE_Kill.value] = 1.2
# self.rewards_table[Action.UNSAFE_KillStar.value] = 2
# self.rewards_table[Action.UNSAFE_Die.value] = -1.2
# self.rewards_table[Action.UNSAFE_GoalZone.value] = 1
# self.rewards_table[Action.UNSAFE_Finish.value] = 2