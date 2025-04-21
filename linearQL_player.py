import numpy as np
from player import Player
from board import UltimateTicTacToeBoard
from collections import defaultdict
import pickle
import os


class LinearQPlayer(Player):
    def __init__(self, id, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.9999, 
                 decay_rate=0.999, load_weights=None) -> None:
        """
        initializes instance of player class

        params:
            learning_rate: how quickly the agent learns
            discount_factor: importance of future rewards
            exploration_rate: exploration vs exploitation
            decay_rate: how quickly epsilon decreases over time

        """
        super().__init__(id)
        self.learning_rate = learning_rate     
        self.discount_factor = discount_factor 
        self.exploration_rate = exploration_rate
        self.decay_rate = decay_rate              
        
        # initialize Q-values dictionary
        self.q_values = defaultdict(float)
        self.weights = self.initialize_weights()

        # load pre-trained weights if provided
        if load_weights and os.path.exists(load_weights):
            self.load_model(load_weights)
            
        # track for training
        self.previous_state = None
        self.previous_action = None
        self.is_training = True

    def initialize_weights(self):
        """initialize feature weights for linear function approximation"""
        
        return {
            'win_subgrid': 10.0,       # value for winning a subgrid
            'block_subgrid': 6.0,      # value for blocking opponent from winning a subgrid
            'center_subgrid': 3.0,     # value for taking center of a subgrid
            'corner_subgrid': 2.0,     # value for taking corner of a subgrid
            'edge_subgrid': 1.0,       # value for taking edge of a subgrid
            'win_game': 100.0,         # value for making a move that wins the game
            'force_subgrid': 5.0,      # value for forcing opponent to play in disadvantageous position
            'open_lines': 4.0,         # value for creating open lines
            'block_lines': 3.0         # value for blocking opponent's lines
        }

    def extract_features(self, board, subgrid, pos):
        """extract features from the board state for a given move (subgrid, pos)"""
        features = defaultdict(float)
        
        # copy board to simulate the move without changing the original
        sim_board = UltimateTicTacToeBoard(init_state=board.get_str_state())
        
        # get opponent id
        opponent_id = 3 - self.id 
        
        # check if move wins a subgrid
        old_subgrid_state = sim_board.subgrids[subgrid].state.copy()
        sim_board.subgrids[subgrid].state[pos] = self.id
        
        # check if this move wins the subgrid
        if sim_board.subgrids[subgrid].check_win(self, pos):
            features['win_subgrid'] = 1.0
            
            # check if this also wins the game
            old_game_state = sim_board.state.copy()
            sim_board.state[subgrid] = self.id
            if sim_board.check_win(self, subgrid):
                features['win_game'] = 1.0
            sim_board.state = old_game_state.copy()
        
        # check if this move blocks opponent from winning subgrid
        sim_board.subgrids[subgrid].state = old_subgrid_state.copy()
        sim_board.subgrids[subgrid].state[pos] = opponent_id
        if sim_board.subgrids[subgrid].check_win(Player(opponent_id), pos):
            features['block_subgrid'] = 1.0
        
        # reset for other feature checks
        sim_board.subgrids[subgrid].state = old_subgrid_state.copy()
        
        n = sim_board.grid_size
        # center
        if pos == (n//2, n//2):  
            features['center_subgrid'] = 1.0
        # corner
        elif pos[0] in (0, n-1) and pos[1] in (0, n-1):  
            features['corner_subgrid'] = 1.0
        # edge
        else: 
            features['edge_subgrid'] = 1.0
        
        # count lines where we can still win
        open_lines = 0
        blocking_lines = 0
        
        # check rows, columns, and diagonals in the subgrid
        for i in range(n):
            # check rows
            row = sim_board.subgrids[subgrid].state[i, :]
            if np.count_nonzero(row == opponent_id) == 0:
                open_lines += 1
            if np.count_nonzero(row == self.id) == 0:
                blocking_lines += 1
                
            # check cols
            col = sim_board.subgrids[subgrid].state[:, i]
            if np.count_nonzero(col == opponent_id) == 0:
                open_lines += 1
            if np.count_nonzero(col == self.id) == 0:
                blocking_lines += 1
        
        # check diagonals
        diag1 = np.array([sim_board.subgrids[subgrid].state[i, i] for i in range(n)])
        diag2 = np.array([sim_board.subgrids[subgrid].state[i, n-1-i] for i in range(n)])
        
        if np.count_nonzero(diag1 == opponent_id) == 0:
            open_lines += 1
        if np.count_nonzero(diag1 == self.id) == 0:
            blocking_lines += 1
            
        if np.count_nonzero(diag2 == opponent_id) == 0:
            open_lines += 1
        if np.count_nonzero(diag2 == self.id) == 0:
            blocking_lines += 1
        
        # normalize and add to features
        max_lines = n * 2 + 2  # rows + cols + diags
        features['open_lines'] = open_lines / max_lines
        features['block_lines'] = blocking_lines / max_lines
        
        return features

    def calculate_q_value(self, features):
        """calculate the Q-value based on features and weights"""
        q_value = 0
        for feature, value in features.items():
            q_value += self.weights.get(feature, 0) * value
        return q_value

    def get_valid_moves(self, board):
        """get all valid moves for the current board state"""
        valid_moves = []
        subgrid = board.curr_subgrid
        
        if subgrid is None:  
            for i in range(board.grid_size):
                for j in range(board.grid_size):
                    if board.state[i, j] == 0:  
                        for x in range(board.grid_size):
                            for y in range(board.grid_size):
                                if board.subgrids[i, j].state[x, y] == 0:  
                                    valid_moves.append(((i, j), (x, y)))
        else:  
            for i in range(board.grid_size):
                for j in range(board.grid_size):
                    if board.subgrids[subgrid].state[i, j] == 0:  
                        valid_moves.append((subgrid, (i, j)))
        
        return valid_moves

    def choose_action(self, board):
        """choose action based on epsilon-greedy policy"""
        valid_moves = self.get_valid_moves(board)
        
        # explore
        if np.random.random() < self.exploration_rate and self.is_training:
            return valid_moves[np.random.randint(0, len(valid_moves))]
        
        # exploit
        best_value = float('-inf')
        best_moves = []
        
        for subgrid, pos in valid_moves:
            features = self.extract_features(board, subgrid, pos)
            q_value = self.calculate_q_value(features)
            
            if q_value > best_value:
                best_value = q_value
                best_moves = [(subgrid, pos)]
            elif q_value == best_value:
                best_moves.append((subgrid, pos))
        
        # if multiple moves are tied, choose randomly from tied moves
        return best_moves[np.random.randint(0, len(best_moves))]

    def learn(self, current_state, action, reward, next_state, done):
        """update weights based on TD learning"""
        if not self.is_training:
            return
            
        subgrid, pos = action
        features = self.extract_features(current_state, subgrid, pos)
        
        # current Q-value
        current_q = self.calculate_q_value(features)
        
        # calculate target
        if done:
            target = reward
        else:
            # find best next action
            next_valid_moves = self.get_valid_moves(next_state)
            max_next_q = float('-inf')
            
            for next_subgrid, next_pos in next_valid_moves:
                next_features = self.extract_features(next_state, next_subgrid, next_pos)
                next_q = self.calculate_q_value(next_features)
                max_next_q = max(max_next_q, next_q)
            
            if max_next_q == float('-inf'):  
                max_next_q = 0
                
            target = reward + self.discount_factor * max_next_q
        
        # TD update
        td_error = target - current_q
        
        # update weights
        for feature, value in features.items():
            self.weights[feature] += self.learning_rate * td_error * value
    
    def update_exploration_rate(self):
        """decay exploration rate over time"""
        self.exploration_rate *= self.decay_rate
    
    def save_model(self, filename):
        """save the trained weights to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)
    
    def load_model(self, filename):
        """load weights from a file"""
        try:
            with open(filename, 'rb') as f:
                self.weights = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def move(self, board):
        """make a move on the board"""
        current_state = UltimateTicTacToeBoard(init_state=board.get_str_state())
        action = self.choose_action(current_state)
        
        # if in training mode, store state and action for learning
        if self.is_training:
            self.previous_state = current_state
            self.previous_action = action
        
        return action

    def post_move_update(self, board, reward, done):
        """update Q-values after a move has been made"""
        if self.is_training and self.previous_state is not None:
            next_state = UltimateTicTacToeBoard(init_state=board.get_str_state())
            self.learn(self.previous_state, self.previous_action, reward, next_state, done)
            self.update_exploration_rate()