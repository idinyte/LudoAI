import ludopy
import numpy as np
import cv2
import pickle
from DeepQlearning.DeepQAgent import DeepQlearningAgent
from Qlearning.QLearningAgent import QLearningAgent

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
                piece_to_move = agent.update((dice, move_pieces, player_pieces, enemy_pieces), g.players) if player_i == agent.player_i  else move_pieces[np.random.randint(0, len(move_pieces))]
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
    return 100*wins/episodes
    
def start_test(folder, agent):
    episodes = 10000
    #agent = "DeepQlearning/pretrained/20240417161145_2pl_4hl/deep_q_agent_players_2_winrate_0.7933333333333333.pkl"
    agent_pkl = None

    with open(agent, 'rb') as file:
        agent_pkl = pickle.load(file)
    winrate_2 = test(episodes, 2, agent_pkl)
    winrate_3 = test(episodes, 3, agent_pkl)
    winrate_4 = test(episodes, 4, agent_pkl)
    with open(f"{folder}/winrate.txt", 'w') as file:
        file.write(f"{episodes} games have been played {agent}\n")
        file.write(f"winrate_2p={winrate_2}% \n")
        file.write(f"winrate_3p={winrate_3}% \n")
        file.write(f"winrate_4p={winrate_4}%\n")
        
#start_test("DeepQlearning/pretrained/20240422102011_4p_4hl", "DeepQlearning/pretrained/20240422102011_4p_4hl/deep_q_agent_players_4_winrate_0.565.pkl")