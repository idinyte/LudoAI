from enum import Enum
from actionTable import ActionTable

class State(Enum):
    SAFE = 0
    UNSAFE = 1

class Action(Enum):
    MoveOut = 0
    Dice = 1
    Star = 2
    Globe = 3
    DoubleUp = 4
    Kill = 5
    KillStar = 6
    KillMoveOut = 7
    Die = 8
    GoalZone = 9
    Finish = 10

class StateSpace():
    action_table_obj = ActionTable(len(State), len(Action))
    players = [None, None, None, None]
    global_pieces = [[0] * 4 ] * 4
    kill_pos, die_pos = [], []
    star_positions = [5, 12, 18, 25, 31, 38, 44, 51]
    global_globes = [9, 22, 35, 48]
    local_globes = [1]

    def update_action_table(self, state, action, piece_i):
        self.action_table_obj.update_action_table(state, action, piece_i)
   
    def set_global_pieces(self, players):
        for player_i, player in enumerate(players):
            for piece_i, piece in enumerate(player.pieces):
                self.global_pieces[player_i][piece_i] = piece if piece == 0 or piece == 59 else (piece + (13 * player_i)) % 52
                
    def global_pos(self, player_i, piece_i):
        return self.global_pieces[player_i][piece_i]

    def local_pos(self, player_i, piece_i):
        return self.players[player_i].pieces[piece_i]

    def local_to_global_pos(self, player_i, local_pos):
        if local_pos == 0 or local_pos == 59:
            return local_pos

        return (local_pos + (13 * player_i)) % 52

    def is_piece_safe(self, player_i, piece_i):
        is_double = self.players[player_i].pieces.count(self.local_pos(player_i, piece_i)) > 1
        
        return is_double or self.is_at_safe_pos(self.local_pos(player_i, piece_i), self.global_pos(player_i, piece_i))

    def is_at_safe_pos(self, local_postion, global_position):
        return global_position in self.global_globes or local_postion in self.local_globes or local_postion >= 53 or local_postion == 0
    
    def local_globes_global(self):
        return [self.local_to_global_pos(pos) for pos in self.local_globes]

    def get_enemies(self, current_player):
        kill_pos = []
        die_pos = []
        for piece_i in range(4):
            if piece_i == current_player:
                continue

            for enemy_piece_i in range(4):
                global_pos = self.global_pieces[piece_i][enemy_piece_i]
                local_pos = self.players[piece_i].pieces[enemy_piece_i]
                if global_pos in self.global_globes or local_pos in self.local_globes:
                    die_pos.append(global_pos)
                else:
                    kill_pos.append(global_pos)

        my_globes = self.local_globes_global()
        for pos in kill_pos[:]:
            if kill_pos.count(pos) > 1 and pos not in my_globes:
                die_pos.append(pos)
                kill_pos = [x for x in kill_pos if x != pos]

        self.kill_pos, self.die_pos = kill_pos, die_pos
            
    def get_state(self, player_i, piece_i):
        local_pos = self.local_pos(player_i, piece_i)
        is_double = self.players[player_i].pieces.count(local_pos) >= 1
        if is_double or self.is_at_safe_pos(local_pos, self.global_pos(player_i, piece_i)):
            return State.SAFE
        else:
            return State.UNSAFE

    def _move_out_action(self, player_i, current_state, piece_i, dice):
        if self.local_pos(player_i, piece_i) == 0 and dice == 6:
            if self.local_to_global_pos(player_i, 1) in self.kill_pos:
                self.update_action_table(current_state, Action.KillMoveOut, piece_i)
            else:
                self.update_action_table(current_state, Action.MoveOut, piece_i)
            return True
        return False

    def _move_dice_action(self, next_pos_local, current_state, piece_i):
        if next_pos_local <= 57:
            self.update_action_table(current_state, Action.Dice, piece_i)
            return True
        return False

    def _star_action(self, next_pos, current_state, piece_i, can_kill):
        if next_pos in self.star_positions:
            next_star_i = self.star_positions.index(next_pos)
            next_star_pos = self.star_positions[(next_star_i + 1) % len(self.star_positions)]
            self._kill_action(next_star_pos, current_state, piece_i)
            if can_kill or next_star_pos in self.kill_pos:
                self.update_action_table(current_state, Action.KillStar, piece_i)
            elif not self._die_action(next_star_pos, current_state, piece_i):
                self.update_action_table(current_state, Action.Star, piece_i)
            return True
        return False

    def _globe_action(self, next_pos, current_state, piece_i):
        if next_pos in self.global_globes:
            self.update_action_table(current_state, Action.Globe, piece_i)
            return True
        return False

    def _double_up_action(self, next_pos, current_state, piece_i, player_i):
        if self.global_pieces[player_i].count(next_pos) >= 1:
            self.update_action_table(current_state, Action.DoubleUp, piece_i)
            return True
        return False

    def _kill_action(self, next_pos, current_state, piece_i):
        if next_pos in self.kill_pos:
            self.update_action_table(current_state, Action.Kill, piece_i)
            return True
        return False

    def _die_action(self, next_pos, current_state, piece_i):
        if next_pos in self.die_pos:
            self.update_action_table(current_state, Action.Die, piece_i)
            return True
        return False

    def _goal_zone(self, next_pos_local, current_state, piece_i):
        if next_pos_local < 52:
            return False

        if next_pos_local == 57:
            self.update_action_table(current_state, Action.Finish, piece_i)
        else:
            self.update_action_table(current_state, Action.GoalZone, piece_i)

        return True

    def update(self, players, current_player, pieces_to_move, dice):
        self.action_table_obj.reset()
        self.players = players
        self.set_global_pieces(players)
        self.get_enemies(current_player)
        for piece_i in pieces_to_move:
            current_state = self.get_state(current_player, piece_i)
            next_pos = self.global_pos(current_player, piece_i) + dice
            local_pos = self.local_pos(current_player, piece_i)
            next_pos_local = local_pos + dice

            is_moving_out = self._move_out_action(current_player, current_state, piece_i, dice)
            if is_moving_out or (not is_moving_out and local_pos == 0):
                continue
            if self._goal_zone(next_pos_local, current_state, piece_i):
                continue
            if self._die_action(next_pos, current_state, piece_i):
                continue
            can_kill = self._kill_action(next_pos, current_state, piece_i)
            if self._star_action(next_pos, current_state, piece_i, can_kill):
                continue
            if self._globe_action(next_pos, current_state, piece_i):
                continue
            if self._double_up_action(next_pos, current_state, piece_i, current_player):
                continue
            if self._move_dice_action(next_pos_local, current_state, piece_i):
                continue