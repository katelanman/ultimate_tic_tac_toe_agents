import numpy as np
from player import Player
from board import UltimateTicTacToeBoard, check_win
import random
import pickle
import os.path
from tqdm import tqdm

class TabularQPlayer(Player):
    def __init__(self, id, alpha=0.1, gamma=0.9, epsilon=0.999, save_path=None, load=False) -> None:
        """
        q-learning agent for Ultimate Tic Tac Toe
        
        params:
            id : int
            alpha : learning rate
            gamma : discount factor
            epsilon : exploration rate (probability of random action)
            save_path : path to save Q-table
            load : whether to load an existing Q-table
        """
        super().__init__(id)
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon = epsilon 
        self.q_table = {}  
        self.save_path = save_path or f"q_table_player{id}.pkl"
        self.prev_state = None
        self.prev_action = None
        
        # load existing Q-table if specified
        if load and os.path.exists(self.save_path):
            self.load_q_table()
    
    def save_q_table(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_q_table(self):
        with open(self.save_path, 'rb') as f:
            self.q_table = pickle.load(f)
    
    def state_to_key(self, board):
        return board.get_str_state()
    
    def get_valid_actions(self, board):
        valid_actions = []
        
        # determine which subgrids to check
        if board.curr_subgrid is None:
            playable_subgrids = []
            for i in range(board.grid_size):
                for j in range(board.grid_size):
                    if board.state[i, j] == 0 and board.subgrids[i, j].num_empty() > 0:
                        playable_subgrids.append((i, j))
        else:
            # must play in the specified subgrid
            if board.subgrids[board.curr_subgrid].num_empty() > 0:
                playable_subgrids = [board.curr_subgrid]
            else:
                return []
        
        # for each playable subgrid, find all valid positions
        for subgrid in playable_subgrids:
            for i in range(board.grid_size):
                for j in range(board.grid_size):
                    if board.subgrids[subgrid].state[i, j] == 0:
                        valid_actions.append((subgrid, (i, j)))
        
        return valid_actions
    
    def get_q_value(self, state_key, action):
        """get Q-value for a state-action pair"""
        # convert action to a hashable format
        action_key = f"{action[0][0]},{action[0][1]},{action[1][0]},{action[1][1]}"
        
        # initialize state in Q-table if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # initialize action in state's Q-values if needed
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        return self.q_table[state_key][action_key]
    
    def update_q_value(self, state_key, action, new_value):
        """update Q-value for a state-action pair"""
        action_key = f"{action[0][0]},{action[0][1]},{action[1][0]},{action[1][1]}"
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        self.q_table[state_key][action_key] = new_value
    
    def choose_action(self, board, training=False):
        """
        choose an action based on epsilon-greedy policy
        
        parameters:
            board : current game board
            training : whether the agent is in training mode
            
        returns:
            tuple of(subgrid, position) representing the chosen action
        """
        state_key = self.state_to_key(board)
        valid_actions = self.get_valid_actions(board)
        
        # if no valid actions, return None
        if not valid_actions:
            return None
        
        # exploration
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # exploitation
        q_values = [self.get_q_value(state_key, action) for action in valid_actions]
        
        # find actions with maximum Q-value
        max_q = max(q_values)
        max_indices = [i for i, q in enumerate(q_values) if q == max_q]
        
        # if actions are tied, choose randomly among tied moves
        chosen_index = random.choice(max_indices)
        return valid_actions[chosen_index]
    
    def calculate_reward(self, board, result, done):
        """
        calculate reward for the current state
        
        parameters:
            board : current game board
            result : game result (player id for winner, -1 for tie, 0 for ongoing)
            done : whether the game is finished
            
        returns:
            reward value (float)
        """
        if done:
            # win
            if result == self.id:  
                return 1.0
            # tie
            elif result == -1:  
                return 0.2
            # loss
            else:  
                return -1.0
        
        # intermediate rewards based on subgrid wins
        reward = 0.0
        for i in range(board.grid_size):
            for j in range(board.grid_size):
                if board.state[i, j] == self.id:
                    reward += 0.1  # small reward for winning a subgrid
                elif board.state[i, j] not in [0, self.id]:
                    reward -= 0.1  # small penalty for losing a subgrid
        
        return reward
    
    def learn(self, current_state, action, next_state, reward, done):
        """
        update Q-values based on observed transition
        
        parameters:
        -----------
        current_state : current state key
        action : action taken
        next_state : next state key
        reward : reward received
        done : whether the episode is finished
        """
        current_q = self.get_q_value(current_state, action)
        
        if done:
            target = reward
        else:
            # get max Q-value for next state
            next_valid_actions = self.get_valid_actions(
                UltimateTicTacToeBoard(init_state=next_state))
            
            if not next_valid_actions:
                next_max_q = 0
            else:
                next_q_values = [self.get_q_value(next_state, next_action) 
                               for next_action in next_valid_actions]
                next_max_q = max(next_q_values) if next_q_values else 0
            
            target = reward + self.gamma * next_max_q
        
        # update Q-value
        new_q = current_q + self.alpha * (target - current_q)
        self.update_q_value(current_state, action, new_q)
    
    def move(self, board):
        """
        choose a move based on current board state
        
        parameters:
            board : current game board
            
        returns:
            tuple of(subgrid, position) representing the chosen action
        """
        current_state = self.state_to_key(board)
        action = self.choose_action(board)
        
        self.prev_state = current_state
        self.prev_action = action
        
        return action
    
    def update(self, board, result, done):
        """
        update Q-values after a move has been made
        
        parameters:
            board : current game board
            result : game result
            done : whether the game is finished
        """
        if self.prev_state is None or self.prev_action is None:
            return
        
        current_state = self.state_to_key(board)
        reward = self.calculate_reward(board, result, done)
        
        self.learn(self.prev_state, self.prev_action, current_state, reward, done)
    
    def train(self, opponent, episodes=1000, show_progress=True):
        """
        train the agent by playing against an opponent
        
        parameters:
            opponent : opponent player
            episodes : number of episodes to train
            show_progress : whether to show progress bar
        """
        wins = 0
        ties = 0
        losses = 0
        
        episode_range = tqdm(range(episodes)) if show_progress else range(episodes)
        
        for _ in episode_range:
            # initialize game
            board = UltimateTicTacToeBoard()
            players = [self, opponent] if self.id == 1 else [opponent, self]
            current_player_idx = 0
            done = False
            moves_history = []
            
            # play 
            while not done:
                current_player = players[current_player_idx]
                
                current_state = self.state_to_key(board)
                
                # choose and make move
                subgrid, pos = current_player.move(board)
                game_state, result, done = board.subgrid_move(subgrid, current_player, pos)
                
                # store move for learning 
                if current_player == self:
                    next_state = self.state_to_key(board)
                    reward = self.calculate_reward(board, result, done)
                    moves_history.append((current_state, (subgrid, pos), next_state, reward, done))
                
                # switch players
                current_player_idx = 1 - current_player_idx
            
            # learn 
            for state, action, next_state, reward, is_done in moves_history:
                self.learn(state, action, next_state, reward, is_done)
            
            # record game result
            if result == self.id:
                wins += 1
            elif result == -1:
                ties += 1
            else:
                losses += 1
            
            # decay epsilon
            if self.epsilon > 0.01:
                self.epsilon *= 0.9999
        
        # save Q-table after training
        if self.save_path:
            self.save_q_table()
        
        return wins, ties, losses