import numpy as np
import random
from DeepQlearning.rewards import Rewards
from DeepQlearning.stateSpace import Action, StateSpace
from DeepQlearning.DeepQNetwork import DQN
import torch
import torch.optim as optim
import sys

class DeepQlearningAgent(StateSpace):
    state, action = -1, -1
    epsilon, gamma, learning_rate = 0, 0, 0
    replay_memory, max_memory = [], 0
    dqn = None
    device = None
    optimizer = None
    debug = False
    total_reward = 0
    # state_replays, next_state_replays, action_replays, reward_replays, available_action_mask_replays, done_replays = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    def __init__(self, player_i, gamma, learning_rate, epsilon, max_memory = 50000):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.rewards = Rewards(len(Action))
        self.player_i = player_i
        self.max_memory = max_memory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(17, 4).to(self.device)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=1e-4)

    def update(self, obs, players):
        dice, move_pieces, player_pieces, enemy_pieces = obs
        super().update_state_space(players, self.player_i, move_pieces, dice)
        self.state = self.get_state(player_pieces, enemy_pieces, dice)
        self.action = self.select_action(move_pieces)

        return self.action
    
    def get_state(self, player_pieces, enemy_pieces, dice):
        return [*player_pieces, *[piece for pieces in enemy_pieces for piece in pieces.tolist()], dice]

    def select_action(self, move_pieces):
        if random.random() < self.epsilon:
            return move_pieces[random.randint(0, len(move_pieces) - 1)]
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([self.state], dtype=torch.float32).to(self.device)
                available_actions_mask_tensor = torch.tensor(self.available_actions_mask(move_pieces), dtype=torch.float32).to(self.device)
                q_values = self.dqn(state_tensor)
                q_values *= available_actions_mask_tensor
                q_values[torch.isnan(q_values)] = -float('inf')
                move = torch.argmax(q_values).item()
                if self.debug:
                    print(f"q values {q_values}")
                    print(f"move {move}")
                
                return move
            
    def save_weights(self, file_path):
        torch.save(self.dqn.state_dict(), file_path)
            
    # def update_dqn(self, batch_size, debug = False):
    #     self.debug = debug
    #     batch = random.sample(self.replay_memory, min(batch_size, len(self.replay_memory)))
    #     states, actions, rewards, next_states, actions_mask, next_actions_mask, done = zip(*batch)
    #     actions_mask = np.array(actions_mask)
    #     next_actions_mask = np.array(actions_mask)
    #     states = torch.tensor(states, dtype=torch.float32).to(self.device)
    #     actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
    #     rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
    #     next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
    #     actions_mask = torch.tensor(actions_mask, dtype=torch.int64).to(self.device)
    #     next_actions_mask = torch.tensor(next_actions_mask, dtype=torch.int64).to(self.device)
    #     done = torch.tensor(done).to(self.device)

    #     q_values = self.dqn.forward(states, actions_mask)     
    #     q_value_of_actions_taken = q_values.gather(1, actions.unsqueeze(1)) # validate gather function
        
    #     next_q_values = self.dqn.forward(next_states, next_actions_mask)
    #     next_q_values[done] = 0
    
    #     target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1)[0]

    #     self.optimizer.zero_grad()
    #     if self.debug:
    #         print(f"q_values {q_values}")
    #         print(f"q_values of action taken {q_value_of_actions_taken}")
    #         print("actions")
    #         print(actions)
    #         print(actions.unsqueeze(1))
    #         print(q_values)
    #         print(q_value_of_actions_taken)
    #         print(target_q_values)
    #     # sys.exit(0)
    #     self.optimizer.zero_grad()
    #     loss = torch.nn.functional.smooth_l1_loss(target_q_values.unsqueeze(1), q_value_of_actions_taken)
    #     loss.backward()
    #     self.optimizer.step()
        
    # def learn_dqn(self, batch_size, debug = False):
    #     if len(self.state_replays) < batch_size:
    #         return
    #     print(f"debug = {debug}")
    #     self.debug = debug
    #     self.optimizer.zero_grad()
    #     batch = np.random.choice(len(self.state_replays), batch_size, replace=False)
    #     batch_index = np.arange(batch_size, dtype=np.int32)
    #     state_batch = torch.tensor(self.state_replays[batch]).to(self.device)
    #     new_state_batch = torch.tensor(self.next_state_replays[batch]).to(self.device)
    #     print(len(self.state_replays))
    #     print(self.state_replays)
    #     print(self.reward_replays)
    #     reward_batch = torch.tensor(self.reward_replays[batch]).to(self.device)
    #     terminal_batch = torch.tensor(self.done_replays[batch]).to(self.device)

    #     action_batch = self.action_replays[batch]
    #     action_space_batch = self.available_action_mask_replays[batch]
    #     new_action_space_batch = self.available_action_mask_replays[batch]

        
        
    #     #q_eval = self.Q_eval.forward(state_batch, action_space_batch)[batch_index, action_batch]
    #     q_eval = self.dqn.forward(state_batch, action_space_batch)[batch_index, action_batch]
    #     q_next = self.dqn.forward(new_state_batch, new_action_space_batch)
    #     q_next[terminal_batch] = -float("inf")
        
    #     if(min(torch.max(q_next, dim=1)[0]) < -1000000):
    #         print(f"no valid")
    #         print(min(torch.max(q_next, dim=1)[0]))
    #         sys.exit(0)
        
    #     q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        
    #     if self.debug:
    #         print(f"q_values {q_eval}")
    #         print(f"target {q_target}")

    #     #loss = self.dqn.loss(q_target, q_eval).to(self.device)
    #     loss = torch.nn.functional.smooth_l1_loss(q_target, q_eval).to(self.device)
    #     loss.backward()
    #     self.optimizer.step()

        #self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        
    def available_actions_mask(self, move_pieces):
        available_actions_mask = np.full(4, np.nan)
        available_actions_mask[move_pieces] = 1
        return available_actions_mask
        
    # def update_dqn_last_move_only(self, next_state, move_pieces, debug=False):
    #     state = torch.tensor(self.state, dtype=torch.float32).to(self.device)
    #     action = torch.tensor(self.action).to(self.device)
    #     reward = torch.tensor(self.get_reward(), dtype=torch.float32).to(self.device)
    #     next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        
    #     available_actions_mask = torch.zeros(4)
    #     available_actions_mask[move_pieces] = 1
        
    #     q_values = self.dqn(state)#.gather(1, action)
        
    #     next_q_values = self.dqn(next_state) #.max(0)[0].detach()
    #     rewards = np.array([0, 0, 0, 0])
    #     for piece_i in move_pieces:
    #         #print(self.action_table_obj.action_table)
    #         action = int(self.action_table_obj.action_table[piece_i])
    #         rewards[piece_i] = self.rewards.rewards_table[action]
            
    #     rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            
    #     #target_q_value = (reward + self.gamma * next_q_value) * available_actions
    #     #target_q_value = q_values.clone().detach() * available_actions_mask  # Clone q_value tensor
    #     target_q_values = (rewards + self.gamma * next_q_values) * available_actions_mask
    #     if debug:
    #         print(f"q_values {q_values}")
    #         print(f"next_q_values {next_q_values}")
    #         print(f"max next q value {next_q_values.max(0)[0].detach()}")
    #         print(f"target {target_q_values}")
    #         print(f"loss {target_q_values - q_values}")
        
    #     loss = torch.nn.functional.smooth_l1_loss(q_values, target_q_values)
        
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
        
    def update_epsilon(self, epsilon, decay_rate, episode):
        self.epsilon = epsilon * np.exp(-decay_rate*episode)

    def get_reward(self):
        action = self.action_table_obj.action_table[self.action] # convert move piece to action that occured
        reward = self.rewards.rewards_table[int(action)]
        if self.debug:
            print(self.action)
            print(f"reward  {reward}")
            self.print_state_action()
        return reward
    
    def train_dqn(self, batch_size, debug):
        if len(self.replay_memory) < batch_size:
            return 500 # Not enough samples in the replay memory
        self.optimizer.zero_grad()
        self.debug = debug
        # Sample a mini-batch from the replay memory
        transitions = np.random.choice(len(self.replay_memory), batch_size, replace=False)
        batch = [self.replay_memory[idx] for idx in transitions]

        states = torch.tensor([transition['state'] for transition in batch], dtype=torch.float32)
        next_states = torch.tensor([transition['next_state'] for transition in batch], dtype=torch.float32)
        rewards = torch.tensor([transition['reward'] for transition in batch], dtype=torch.float32)
        actions = torch.tensor([transition['action'] for transition in batch], dtype=torch.long)
        dones = torch.tensor([transition['done'] for transition in batch], dtype=torch.float32)

        # Compute Q-values for the current state
        q_values = self.dqn.forward(states)

        # Compute Q-values for the next state
        next_q_values = self.dqn.forward(next_states)

        # Compute target Q-values
        target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1).values * (1 - dones)

        # Mask out unavailable actions
        mask = torch.eye(self.dqn.fc3.out_features)[actions]
        masked_q_values = torch.sum(q_values * mask, dim=1)
        
        # Compute loss
        loss = torch.nn.functional.smooth_l1_loss(masked_q_values, target_q_values.detach())
        torch.nn.utils.clip_grad_value_(self.dqn.parameters(), 100)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def updateReplayMemory(self, reward, next_state, move_pieces, done):
        replay = {
            'state' : self.state,
            'next_state': next_state,
            'action' : self.action,
            'reward': reward,
            'done': done
        }
        if len(self.replay_memory) < self.max_memory:
            self.replay_memory.append(replay)
        else:
            self.replay_memory[np.random.randint(self.max_memory)] = replay
        # if len(self.state_replays) < self.max_memory:
        #     #self.replay_memory.append(replay)
        #     self.state_replays = np.append(self.state_replays, self.state)
        #     self.next_state_replays = np.append(self.next_state_replays, next_state)
        #     self.action_replays = np.append(self.action_replays, self.action)
        #     self.reward_replays = np.append(self.reward_replays, reward)
        #     self.available_action_mask_replays = np.append(self.available_action_mask_replays, self.available_actions_mask(move_pieces))
        #     self.done_replays = np.append(self.done_replays, done)
        # else:
        #     index = np.random.randint(self.max_memory)
        #     self.state_replays[index] = self.state
        #     self.next_state_replays[index] = next_state
        #     self.action_replays[index] = self.action
        #     self.reward_replays[index] = reward
        #     self.available_action_mask_replays[index] = self.available_actions_mask(move_pieces)
        #     self.done_replays[index] = done
        
    def print_state_action(self):
        action_str = ""
        match self.action_table_obj.action_table[self.action]:
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