

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

        # optimizes for centers or corners of subgrids 
        position_weights = [
            [3, 1, 3],
            [1, 5, 1],
            [3, 1, 3]
        ]

        for i in range(board.grid_size):
            for j in range(board.grid_size):
                if board.state[i][j] == self.id:
                    player_positions += 1 * position_weights[i][j]
                elif board.state[i][j] != 0:
                    opponent_positions += 1 * position_weights[i][j]
        
        return player_positions - opponent_positions
    
    def get_children(self, board):
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
        return self.sort_children(all_valid_moves)

    def sort_children(self, all_valid_moves):
        """
        Sort children based on optimal subgrid positions (center > corners > sides)
        """
        moves_values = []
        for move in all_valid_moves:
            subgrid_pos, pos = move
            # Prefer center of subgrids and corners
            center_preference = 5 if pos == (1, 1) else 3 if pos in [(0,1), (1,0), (1,2), (2,1)] else 1
            # Prefer strategic subgrids (center or corners)
            subgrid_preference = 5 if subgrid_pos == (1, 1) else 3 if subgrid_pos in [(0,1), (1,0), (1,2), (2,1)] else 1
            moves_values.append((move, center_preference * subgrid_preference))
        return [move for move, _ in sorted(moves_values, key=lambda x: x[1], reverse=True)]

    def mini_max(self, board, depth, alpha, beta, is_maximizing):
        """
        Minimax algorithm with alpha-beta pruning
        """
        # get hash of board state
        board_state_hash = board.get_str_state()[:-1]

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
        all_valid_moves = self.get_children(board)
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
            valid_moves = self.get_children(board)
            if valid_moves:
                return valid_moves[0]
            raise ValueError("No valid moves available")
            
        return best_move
