

import numpy as np
from player import Player
from board import check_win
import copy

class AlphaBetaMiniMaxPlayer(Player):
    def __init__(self, id, depth, is_maximizing=True) -> None:
        super().__init__(id)
        self.depth = depth
        self.is_maximizing = is_maximizing
        # if alpha beta player is 2 (O), opponent is 1 (X) and vice versa 
        self.opponent_id = 2 if id == 1 else 1
        # transposition table to store visited board states and corresponding utility values 
        self.transposition_table = {}

    def get_utility_val(self, board, is_game_done) -> int:
        """
        Calculates the utility value of current board state
        If game is completed, awards +100 for alpha beta player winning, -100 for opponent winning and 0 for tie 
        Calculates a heuristic value (# of alpha beta player position - # of oppponent player position) on entire board
        """
        if is_game_done:
            winner_id = check_win(board.state)
            if winner_id == self.id:
                return 100
            elif winner_id == self.opponent_id:
                return -100
            else:
                return 0
        
        # calculate a heurisitic value when game is done
        # optimizes for alpha beta agent to occupy more positions than opponent
        player_positions = 0
        opponent_positions = 0

        for i in range(board.grid_size):
            for j in range(board.grid_size):
                if board.state[i][j] == self.id:
                    player_positions += 1 * self.get_position_weights(board, i, j)
                elif board.state[i][j] == self.opponent_id:
                    opponent_positions += 1 * self.get_position_weights(board, i, j)
        
        return player_positions - opponent_positions
    
    def get_position_weights(self, board, i, j) -> int:
        # all positions in a 2x2 are corners 
        if board.grid_size == 2:
            return 3
        max_dim = board.grid_size - 1
        if (i == j):
            # center position 
            if ((i != 0 or i != max_dim)): return 5
            # corner pieces (left diagonal)
            else: return 3
            # corner pieces (right diagonal)
        elif ((i == 0 and j == max_dim) or (i == max_dim and j == 0)):
            return 3
        else:
            return 1

    def get_children(self, board, is_acsending):
        """
        Returns all valid moves based on current board state 
        return type: [((subgrid_pos_x,subgrid_pos_y), (empty_pos_x,empty_pos_y)), ...]
        """
        all_valid_moves = []
        curr_subgrid_pos = board.curr_subgrid
        # get all empty positions in current subgrid 
        if (curr_subgrid_pos):
            curr_subgrid = board.subgrids[curr_subgrid_pos].state
            all_empty_pos = tuple(np.argwhere(curr_subgrid == 0).tolist())
            for empty_pos in all_empty_pos:
                all_valid_moves.append((curr_subgrid_pos, tuple(empty_pos)))
        # get all empty positions in entire board
        else:
            for rowIndex, row in enumerate(board.subgrids):
                for colIndex, subgrid in enumerate(row): 
                    if subgrid.playable:
                        all_empty_pos = np.argwhere(subgrid.state == 0)
                        for empty_pos in all_empty_pos:
                            all_valid_moves.append(((rowIndex, colIndex), tuple(empty_pos.tolist())))
        return self.sort_children(all_valid_moves, is_acsending, board)

    def sort_children(self, all_valid_moves, is_acsending, board):
        """
        Sort children based on optimal subgrid positions (center > corners > sides)
        Ordering depends on is_acsending flag where True sorts moves from most optimal to least 
        """
        moves_values = []
        for move in all_valid_moves:
            subgrid_pos, pos = move
            # Weight for individual subgrid 
            preference = self.get_position_weights(board, pos[0], pos[1])
            # Weight for board
            subgrid_preference = self.get_position_weights(board, subgrid_pos[0], subgrid_pos[1])
            moves_values.append((move, preference * subgrid_preference))
        return [move for move, _ in sorted(moves_values, key=lambda x: x[1], reverse=is_acsending)]

    def mini_max(self, board, depth, alpha, beta, is_maximizing):
        """
        Minimax algorithm with alpha-beta pruning
        """
        # get hash of board state
        board_state_hash = board.get_str_state()

        if board_state_hash in self.transposition_table:
            utility_value, stored_depth, best_move_stored = self.transposition_table[board_state_hash]
            if stored_depth >= depth:
                return utility_value, best_move_stored

        # check if board state is a terminal node (game is done or depth is reached)
        is_game_done = check_win(board.state) != 0 or board.num_empty() == 0

        if depth == 0 or is_game_done:
            utility_value = self.get_utility_val(board, is_game_done)
            self.transposition_table[board_state_hash] = (utility_value, depth, None)
            return utility_value, None
        
        # get all children of current board state 
        all_valid_moves = self.get_children(board, is_maximizing)
        if not all_valid_moves:
            return 0, None 
        
        best_move = None
        
        # maximizing the utility value
        if is_maximizing:
            max_eval = float('-inf')
            for move in all_valid_moves:

                board_copy = copy.deepcopy(board)
                subgrid_pos, pos = move
                
                player = Player(self.id)
                try:
                    board_copy.subgrid_move(subgrid_pos, player, pos)
                    eval_score, _ = self.mini_max(board_copy, depth - 1, alpha, beta, False)
                    
                    # updates max and best score if eval score is better (larger than current max)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_move = move
                    
                    alpha = max(alpha, eval_score)
                    # prune 
                    if beta <= alpha:
                        break 
                # continues for invalid moves 
                except ValueError:
                    continue
            self.transposition_table[board_state_hash] = (max_eval, depth, best_move)
            return max_eval, best_move
        # minimizing the utility value
        else:
            min_eval = float('inf')
            for move in all_valid_moves:
                board_copy = copy.deepcopy(board)
                subgrid_pos, pos = move
                
                opponent = Player(self.opponent_id)
                try:
                    board_copy.subgrid_move(subgrid_pos, opponent, pos)
                    eval_score, _ = self.mini_max(board_copy, depth - 1, alpha, beta, True)
                    
                    # updates min and best score if eval score is better (smaller than current min)
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_move = move
                    
                    beta = min(beta, eval_score)
                    # prune 
                    if beta <= alpha:
                        break
                # continues for invalid moves 
                except ValueError:
                    continue
            self.transposition_table[board_state_hash] = (min_eval, depth, best_move)  
            return min_eval, best_move
    
    def move(self, board):
        """
        Finds best move using mini max algorithm or returns first valid move if none is found 
        """
        _, best_move = self.mini_max(board, self.depth,float('-inf'), float('inf'), True)
        # if no best move, first valid move is made 
        if best_move is None:
            valid_moves = self.get_children(board, True)
            if valid_moves:
                return valid_moves[0]
            raise ValueError("No valid moves available")
            
        return best_move
