import ludopy
import numpy as np
import cv2
import QLearningAgent

def train(episodes, players, learning_rate, discount_factor, epsilon, decay, render_environment = False):
    assert 2 <= players <= 4, "There must be between 2 and 4 players"

    if players == 2:
        g = ludopy.Game(ghost_players=[1,3])
    elif players == 3:
        g = ludopy.Game(ghost_players=[2])
    elif players == 4:
        g = ludopy.Game(ghost_players=[])
    
    agent = QLearningAgent(0, learning_rate, discount_factor)
    
    for episode in range(episodes):
        there_is_a_winner = False
        g.reset()
        while not there_is_a_winner:
            obs =  g.get_observation()
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = obs

            if len(move_pieces):
                piece_to_move = agent.update(dice, move_pieces, g.players) if player_i == agent.player_idx  else move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1

            new_obs = g.answer_observation(piece_to_move)
            _, _, _, _, player_is_a_winner, there_is_a_winner = new_obs
            
            if render_environment:
                enviroment_image_rgb = g.render_environment() # RGB image of the enviroment
                enviroment_image_bgr = cv2.cvtColor(enviroment_image_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("Enviroment", enviroment_image_bgr)
                cv2.waitKey(50)

episode = 300
players = 2
learning_rate = 0.4
discount_factor = 0.4 # importance of future rewards
epsilon = 0.99
decay = 0.01
train(100, 4, learning_rate, discount_factor, epsilon, decay)


cv2.destroyAllWindows()

# print("Saving history to numpy file")
# g.save_hist(f"game_history.npz")
# print("Saving game video")
# g.save_hist_video(f"game_video.mp4")