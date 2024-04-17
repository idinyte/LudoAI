import ludopy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Qlearning.QLearningAgent import QLearningAgent
import pickle

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

def train(episodes, players, learning_rate, discount_factor, epsilon, decay, debug_actions = False):
    assert 2 <= players <= 4, "There must be between 2 and 4 players"

    if players == 2:
        g = ludopy.Game(ghost_players=[1,3])
    elif players == 3:
        g = ludopy.Game(ghost_players=[2])
    elif players == 4:
        g = ludopy.Game(ghost_players=[])
    
    agent = QLearningAgent(0, learning_rate, discount_factor, epsilon)
    
    wins = 0
    epsilon_list = []

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
                
        if g.first_winner_was == agent.player_i:
            wins += 1

        agent.q_learning.update_epsilon(epsilon, decay, episode)
        epsilon_list.append(agent.q_learning.epsilon_greedy)
        
        if episode % 100 == 0:
            print(f"episode: {episode}")
            print(f"win rate: {wins}%")
            wins = 0

    # save the agent
    with open('Qlearning/pretrained/q_learning_agent.pkl', 'wb') as file:
        pickle.dump(agent, file)
    plot_epsilon(epsilon_list)

episodes = 1500
players = 2
learning_rate = 0.2
discount_factor = 0.4 # importance of future rewards
epsilon = 0.99
decay = 0.005
debug_actions = False
train(episodes, players, learning_rate, discount_factor, epsilon, decay, debug_actions)


cv2.destroyAllWindows()

# print("Saving history to numpy file")
# g.save_hist(f"game_history.npz")
# print("Saving game video")
# g.save_hist_video(f"game_video.mp4")