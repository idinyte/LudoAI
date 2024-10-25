import ludopy
import numpy as np
import cv2
import os
import fnmatch
import pickle
import copy

def test(episodes, players, q_agent, dq_agent, q_agent2=None, dq_agent2=None):
    assert 2 <= players <= 4, "There must be between 2 and 4 players"

    if players == 2:
        g = ludopy.Game(ghost_players=[2,3])
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
                piece_to_move = None
                if player_i == q_agent.player_i:
                    piece_to_move = q_agent.update(dice, move_pieces, g.players)
                elif  player_i == dq_agent.player_i:
                    piece_to_move = dq_agent.update((dice, move_pieces, player_pieces, enemy_pieces), g.players)
                elif q_agent2 != None and player_i == q_agent2.player_i:
                    piece_to_move = q_agent2.update(dice, move_pieces, g.players)
                elif dq_agent2 != None and player_i == dq_agent2.player_i:
                    piece_to_move = dq_agent2.update((dice, move_pieces, player_pieces, enemy_pieces), g.players)
                #move_pieces[np.random.randint(0, len(move_pieces))]
            else:
                piece_to_move = -1
            
            new_obs = g.answer_observation(piece_to_move)
            _, _, _, _, player_is_a_winner, there_is_a_winner = new_obs
                
        if g.first_winner_was == q_agent.player_i or (q_agent2 != None and g.first_winner_was == q_agent2.player_i):
            wins += 1

        
        if episode % 100 == 0:
            print(f"episode: {episode}")
            print(f"win rate: {100*wins/episode}%")
            
    print(f"final win rate: {100*wins/episodes}%")
    return 100*wins/episodes

def full_test(workdir, q_agent, dq_agent, episodes):
    q_agent_pkl=None
    dq_agent_pkl=None
    with open(f"{workdir}/{q_agent}", 'rb') as file:
        q_agent_pkl = pickle.load(file)
    with open(f"{workdir}/{dq_agent}", 'rb') as file:
        dq_agent_pkl = pickle.load(file)
    q_agent_pkl_copy = copy.deepcopy(q_agent_pkl)
    dq_agent_pkl_copy = copy.deepcopy(dq_agent_pkl)
    q_agent_pkl.player_i = 0
    dq_agent_pkl.player_i = 1
    q_agent_pkl_copy.player_i = 2
    dq_agent_pkl_copy.player_i = 3
    winrate_2 = test(episodes, 2, q_agent_pkl, dq_agent_pkl)
    winrate_4 = test(episodes, 4, q_agent_pkl, dq_agent_pkl, q_agent_pkl_copy, dq_agent_pkl_copy)
    with open(f"{workdir}/winrate.txt", 'w') as file:
        file.write(f"{episodes} games have been playd {q_agent_pkl}\n")
        file.write(f"winrate_2p={winrate_2}% \n")
        file.write(f"winrate_4p={winrate_4}%\n")


# path = 'Qlearning/pretrained/20240525124255'
# directories = [d for d in os.listdir(path)]
# for dir in directories:
#     workdir = path + "/" + dir
#     agent = [f for f in os.listdir(workdir) if fnmatch.fnmatch(f, '*.pkl')][0]
#     full_test(workdir, agent, 1000)


full_test("AgentArena/match1", "q_agent_players_4_winrate_0.64.pkl", "deep_q_agent_players_4_winrate_0.565.pkl", 10000)

cv2.destroyAllWindows()

# print("Saving history to numpy file")
# g.save_hist(f"game_history.npz")
# print("Saving game video")
# g.save_hist_video(f"game_video.mp4")