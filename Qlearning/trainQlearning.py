import ludopy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Qlearning.QLearningAgent import QLearningAgent
import pickle
import os
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
    
def save_arr_to_txt(file_path, data):
    with open(file_path, 'w') as file:
        for value in data:
            file.write(str(value) + '\n')
            
def save_data(save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(f"{save_folder}/q_agent_players_{players}_winrate_{sum(wins_list[-1000:])/1000}.pkl", 'wb') as file:
        pickle.dump(agent, file)
    save_arr_to_txt(f"{save_folder}/wins_list.txt", wins_list)
    save_arr_to_txt(f"{save_folder}/epsilon_list.txt", epsilon_list)

def plot_epsilon(list):
    plt.plot(list)
    plt.xlabel('episode')
    plt.ylabel('epsilon')
    plt.title('Change of epsilon')
    plt.show()

def train(agent, episodes, save_folder, players, learning_rate, discount_factor, epsilon, decay, debug_actions = False):
    global epsilon_list, wins_list
    assert 2 <= players <= 4, "There must be between 2 and 4 players"

    if players == 2:
        g = ludopy.Game(ghost_players=[1,3])
    elif players == 3:
        g = ludopy.Game(ghost_players=[2])
    elif players == 4:
        g = ludopy.Game(ghost_players=[])

    # debug
    action_str = ""
    for episode in range(1, episodes + 1):
        there_is_a_winner = False
        g.reset()
        while not there_is_a_winner:
            obs =  g.get_observation()
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = obs

            if len(move_pieces):
                piece_to_move = agent.update(dice, move_pieces, g.players) if player_i == agent.player_i  else move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1

            if episode > 1000 and debug_actions and player_i == agent.player_i and piece_to_move != -1:
                action_str = agent.print_state_action()
                print("before")
                show_environment(f"Enviroment before action", g, scale=0.75)
            
            new_obs = g.answer_observation(piece_to_move)
            _, _, _, _, player_is_a_winner, there_is_a_winner = new_obs
            
            if episode > 2000 and debug_actions and player_i == agent.player_i and piece_to_move != -1:
                print("after")
                show_environment(f"Enviroment after action", g, scale=0.75)
                
            if agent.player_i == player_i and piece_to_move != -1:
                agent.reward(g.players, [piece_to_move])
                
        # gather data
        agent.q_learning.update_epsilon(epsilon, decay, episode)
        epsilon_list.append(agent.q_learning.epsilon_greedy)
        wins_list.append(int(g.first_winner_was == agent.player_i))
        if episode % 100 == 0:
            print(f"episode: {episode}")
            print(f"epsilon: {agent.q_learning.epsilon_greedy}")
            print(f"win rate: {sum(wins_list[-100:])}%")

    save_data(save_folder)

save_folder = f"Qlearning/pretrained/{datetime.now().strftime('%Y%m%d%H%M%S')}"
epsilon_list, wins_list = [], []
episodes = 1
players = 4
learning_rates = [0.1, 0.01, 0.001]
discount_factors = [0.95, 0.9, 0.8, 0.4] # importance of future rewards
epsilon = 0.99
decay = 0.003

debug_actions = False
for learning_rate in learning_rates:
    for discount_factor in discount_factors:
        agent = QLearningAgent(0, learning_rate, discount_factor, epsilon)
        actual_save_folder = save_folder + f"/lr-{learning_rate}-df-{discount_factor}"
        train(agent, episodes, actual_save_folder, players, learning_rate, discount_factor, epsilon, decay, debug_actions)

cv2.destroyAllWindows()

# print("Saving history to numpy file")
# g.save_hist(f"game_history.npz")
# print("Saving game video")
# g.save_hist_video(f"game_video.mp4")