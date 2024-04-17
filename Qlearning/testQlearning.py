import ludopy
import numpy as np
import cv2
import pickle

def test(episodes, players, agent):
    assert 2 <= players <= 4, "There must be between 2 and 4 players"

    if players == 2:
        g = ludopy.Game(ghost_players=[1,3])
    elif players == 3:
        g = ludopy.Game(ghost_players=[2])
    elif players == 4:
        g = ludopy.Game(ghost_players=[])
    
    wins = 0

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
            
            new_obs = g.answer_observation(piece_to_move)
            _, _, _, _, player_is_a_winner, there_is_a_winner = new_obs
                
        if g.first_winner_was == agent.player_i:
            wins += 1

        
        if episode % 100 == 0:
            print(f"episode: {episode}")
            print(f"win rate: {100*wins/episode}%")
            
    print(f"final win rate: {100*wins/episodes}%")

episodes = 10000
players = 2
with open('pretrained/q_learning_agent.pkl', 'rb') as file:
    agent = pickle.load(file)
test(episodes, players, agent)


cv2.destroyAllWindows()

# print("Saving history to numpy file")
# g.save_hist(f"game_history.npz")
# print("Saving game video")
# g.save_hist_video(f"game_video.mp4")