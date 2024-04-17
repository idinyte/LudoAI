import ludopy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from DeepQlearning.DeepQAgent import DeepQlearningAgent
import pickle
import torch
from datetime import datetime

def show_environment(title, g, scale):
    enviroment_image_rgb = g.render_environment() # RGB image of the enviroment
    enviroment_image_bgr = cv2.cvtColor(enviroment_image_rgb, cv2.COLOR_RGB2BGR)
    height, width = enviroment_image_bgr.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_environment = cv2.resize(enviroment_image_bgr, (new_width, new_height))
    cv2.imshow(title, resized_environment)
    cv2.waitKey(0)

def plot_epsilon(list):
    plt.plot(list)
    plt.xlabel('episode')
    plt.ylabel('epsilon')
    plt.title('Change of epsilon')
    plt.show()
    
def save_arr_to_txt(file_path, data):
    with open(file_path, 'w') as file:
        for value in data:
            file.write(str(value) + '\n')
    
def train(agent, episodes, players, learning_rate, discount_factor, epsilon, decay, save_folder, debug_actions = False, debug_start_episode = 0):
    assert 2 <= players <= 4, "There must be between 2 and 4 players"

    if players == 2:
        g = ludopy.Game(ghost_players=[1,3])
    elif players == 3:
        g = ludopy.Game(ghost_players=[2])
    elif players == 4:
        g = ludopy.Game(ghost_players=[])
    
    epsilon_list = []
    loss_list = []
    total_reward_list = []
    wins_list = []

    # debug
    action_str = ""
    
    for episode in range(1, episodes + 1):
        there_is_a_winner = False
        g.reset()
        while not there_is_a_winner:
            obs =  g.get_observation()
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = obs

            if len(move_pieces):
                piece_to_move = agent.update((dice, move_pieces, player_pieces, enemy_pieces), g.players) if player_i == agent.player_i  else move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1

            if episode > debug_start_episode and debug_actions and player_i == agent.player_i and piece_to_move != -1:
                action_str = agent.print_state_action()
                show_environment(f"Enviroment before action", g, scale=0.75)
            
            new_obs = g.answer_observation(piece_to_move)
            _, new_move_pieces, new_player_pieces, new_enemy_pieces, player_is_a_winner, there_is_a_winner = new_obs
            
            # if episode > debug_start_episode and debug_actions and player_i == agent.player_i and piece_to_move != -1:
            #     print("after")
            #     show_environment(f"Enviroment after action", g, scale=0.75)
                
            if agent.player_i == player_i and piece_to_move != -1:
                reward = agent.get_reward()
                agent.total_reward += reward
                next_state = agent.get_state(new_player_pieces, new_enemy_pieces, dice)
                agent.updateReplayMemory(reward, next_state, move_pieces, int(there_is_a_winner))
                #agent.update_dqn(64)
                
                
        if g.first_winner_was == agent.player_i:
            wins_list.append(1)

        loss = agent.train_dqn(3000, episode > debug_start_episode and debug_actions)
        loss_list.append(loss)
        agent.update_epsilon(epsilon, decay, episode)
        epsilon_list.append(agent.epsilon)
        total_reward_list.append(agent.total_reward)
        agent.total_reward = 0
        if episode % 100 == 0:
            print(f"episode: {episode}")
            print(f"epsilon: {agent.epsilon}")
            print(f"win rate: {sum(wins_list[-100:])/100}%")
            print(f"avg reward {sum(total_reward_list[-100:])/100}")
            print(f"avg loss {sum(loss_list[-100:])/100}")

    # save the agent
    with open(f"{save_folder}/deep_q_agent_{players}_players_{sum(wins_list[-300:])/300}_winrate.pkl", 'wb') as file:
        pickle.dump(agent, file)
    agent.save_weights(f"{save_folder}/weights_{sum(loss_list[-100:])/100}_loss.pth")
    save_arr_to_txt(f"{save_folder}/wins_list.txt", wins_list)
    save_arr_to_txt(f"{save_folder}/epsilon_list.txt", epsilon_list)
    save_arr_to_txt(f"{save_folder}/loss_list.txt", loss_list)    
    #plot_epsilon(epsilon_list)

episodes = 1000
players = 2
learning_rate = 0.001
discount_factor = 0.5 # importance of future rewards
epsilon = 0.99
decay = 0.002
debug_actions = False
debug_start_episode = 50000
agent = DeepQlearningAgent(0, discount_factor, learning_rate, epsilon)

save_folder = f"DeepQlearning/pretrained/{datetime.now().strftime('%Y%m%d%H%M%S')}"
train(agent, episodes, players, learning_rate, discount_factor, epsilon, decay, save_folder, debug_actions, debug_start_episode)


cv2.destroyAllWindows()

# print("Saving history to numpy file")
# g.save_hist(f"game_history.npz")
# print("Saving game video")
# g.save_hist_video(f"game_video.mp4")