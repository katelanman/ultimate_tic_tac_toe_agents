import numpy as np
from player import Player
from board import UltimateTicTacToeBoard, check_win
from collections import defaultdict
from tqdm import tqdm
import time

def check_open(line, player):
    """ checks if a tic-tac-toe line could be won by a given player """
    return len(np.where((line != 0) & (line != player))) == 0

class MCTSPlayer(Player):
    def __init__(self, id, exploration_weight=1, calculation_time=1, n_rollouts=1) -> None:
        super().__init__(id)
        self.exploration_weight = exploration_weight
        self.calculation_time = calculation_time # amount of time to calculate a move
        self.n_rollouts = n_rollouts # number of iterations to run per rollout phase

        self.explored = defaultdict(dict) # {state: {children: [], wins: w, visits: n}}
        self.curr_path = []
        self.game_size = None

    def count_open_lines(self, board):
        """ the number of lines open to the player """
        open = 0

        # check rows and cols
        for i in range(board.grid_size):
            row = board.state[i,:]
            col = board.state[:,i]

            if check_open(row):
                open += 1
            if check_open(col):
                open += 1

        # check diagonals
        front_diag = np.array([board.state[i,i] for i in range(board.grid_size)])
        back_diag = np.array([board.state[board.grid_size - i:i] for i in range(board.grid_size)])
        if check_open(front_diag): open += 1
        if check_open(back_diag): open += 1

        return open

    def OLA(self, board):
        """ evaluation function based on the number of open lines available at state t """
        pass
    
    def get_children(self, state, next_player):
        """ get all possible next moves for a given state """
        children = []
        n = self.game_size

        # get playable subgrid state 
        subgrid = int(state[n**4:])
        if subgrid != 0:
            start_i, end_i = [(subgrid - 1) * (n**2), subgrid * (n**2)]
        else:
            start_i, end_i = [0, n**4]
            
        for i in range(start_i, end_i):
            # if square is empty, replace with player id
            if state[i] == "0":
                next_state = state[:i] + str(next_player) + state[i + 1:(n**4)]
                next_grid = i % (n**2) + 1

                # check if next_grid is playable
                next_subgrid = next_state[(next_grid - 1) * (n**2): next_grid * (n**2)]
                next_won = check_win(np.array([[int(next_subgrid[i:i+n][j]) for j in range(n)] for i in range(0,n**2,n)]))
                if "0" not in next_subgrid or next_won != 0:
                    next_grid = 0

                children.append(next_state + str(next_grid))

        return children

    def add_node(self, node):
        """ add node to the tree """  
        if node not in self.explored:    
            next_player = self.id if len(self.curr_path) % 2 == 1 else self.id % 2 + 1  
            children = self.get_children(node, next_player) # curr player corresponds to depth in tree
            self.explored[node] = {"children": children,
                                    "unexplored_children": children.copy(),
                                    "wins": 0,
                                    "visits": 0}
    
    def UCT(self, node):
        """ upper confidence applied to trees formula """
        # if no children have been explored, return random node
        if len(self.explored[node]['children']) - len(self.explored[node]['unexplored_children']) == 0:
            return np.random.choice(self.explored[node]['children'])

        N = self.explored[node]['visits']
        max_bound = -1e9999
        best_node = None

        for child in self.explored[node]['children']:
            if child not in self.explored:
                self.add_node(child)
                return child

            # calculate UCB formula for each child node 
            X = self.explored[child]['wins']
            n = self.explored[child]['visits']
            bound = (X / n) + self.exploration_weight * ((np.log(N) / n) ** 0.5)

            if bound > max_bound:
                max_bound = bound # best value seen so far
                best_node = child 

        return best_node 
    
    def update_node(self, node, w):
        """ update node wins and visits """
        if node not in self.explored:
            self.add_node(node)

        self.explored[node]["wins"] += w
        self.explored[node]["visits"] += 1

    def rollout(self, start_state):
        """ run a simulated playout from a given start state"""
        # simulation
        sim_board = UltimateTicTacToeBoard(init_state=start_state, grid_size=self.game_size)
        player = Player(self.id)
        opponent = Player(self.id % 2 + 1)
        
        done = check_win(sim_board.state) != 0
        result = done
        curr_player = player if len(self.curr_path) % 2 == 1 else opponent
        while not done:
            subgrid, move = player.move(sim_board)
            game_state, result, done = sim_board.subgrid_move(subgrid, curr_player, move)

            # next player
            curr_player = opponent if curr_player == player else player

        sim_board.reset()
        win = 1 if result == self.id else 0 if result == 0 else -1

        # backprop
        for node in self.curr_path:
            if node not in self.explored:
                self.add_node(node)

            self.explored[node]['wins'] += win
            self.explored[node]['visits'] += 1

    def pick_unvisited(self, parent_node):
        """ expand next child node """
        if not self.explored[parent_node]['unexplored_children']:
            return None
        
        selected = self.explored[parent_node]['unexplored_children'].pop()
        self.curr_path.append(selected)
        self.add_node(selected)

        return selected

    def run_simulation(self, current):
        self.curr_path = [current]
        if current not in self.explored:
            self.add_node(current)

        # check if node is terminal
        if len(self.explored[current]['children']) == 0:
            return current

        # if current is not a leaf node
        unexplored = self.explored[current]['unexplored_children']
        while len(unexplored) == 0:
            # check if node is terminal
            if len(self.explored[current]['children']) == 0:
                return current
             
            current = self.UCT(current)
            unexplored = self.explored[current]['unexplored_children']
            self.curr_path.append(current)

        # if node has been rolled out
        if self.explored[current]["visits"] != 0:
            current = self.pick_unvisited(current)

        for _ in range(self.n_rollouts):
            self.rollout(current)
    
    def move(self, board):
        """ select move using MCTS """
        if self.game_size is None:
            self.game_size = board.grid_size

        self.explored = defaultdict(dict)
        state = board.get_str_state()
        self.add_node(state)

        # run simulations for calculation time
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            self.run_simulation(state)
        
        # calculate best move
        pos = self.UCT(state)
        idx = int([i for i in range(self.game_size ** 4) if state[i] != pos[i]][0])
        inner_pos = idx % (self.game_size ** 2)
        outer_pos = idx // (self.game_size ** 2)

        return tuple((outer_pos // self.game_size, outer_pos % self.game_size)), \
                tuple((inner_pos // self.game_size, inner_pos % self.game_size))

