import numpy as np
from player import Player
from board import UltimateTicTacToeBoard, check_win
from collections import defaultdict
from tqdm import tqdm

def check_open(line, player):
    """ checks if a tic-tac-toe line could be won by a given player """
    return len(np.where((line != 0) & (line != player))) == 0

class MCTSPlayer(Player):
    def __init__(self, id, exploration_weight=1) -> None:
        super().__init__(id)
        self.exploration_weight = exploration_weight
        self.explored = defaultdict(dict) # {state: {children: [], wins: w, visits: n}}
        self.n = 0
        self.curr_path = []

        # add start node to tree
        # self.start_node = '0' * 82
        # self.curr_path = [self.start_node]
        # self.add_node(self.start_node)

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
        # TODO: len of state rep is currently hard-coded to fit 3x3 ultimate board (9x9 grid) 
        children = []

        # get playable subgrid state 
        subgrid = int(state[81:])
        if subgrid != 0:
            start_i, end_i = [(subgrid - 1) * 9, subgrid * 9]
        else:
            start_i, end_i = [0, 81]
            
        for i in range(start_i, end_i):
            # if square is empty, replace with player id
            if state[i] == "0":
                next_state = state[:i] + str(next_player) + state[i + 1:81]
                next_grid = i % 9 + 1

                # check if next_grid is playable
                next_subgrid = next_state[(next_grid - 1) * 9: next_grid * 9]
                if "0" not in next_subgrid or check_win(np.array([[next_subgrid[i:i+3][j] for j in range(3)] for i in range(0,9,3)])):
                    next_grid = 0

                children.append(next_state + str(next_grid))

        return children

    def add_node(self, node):
        """ add node to the tree """      
        next_player = self.id if len(self.curr_path) % 2 == 1 else self.id % 2 + 1    
        children = self.get_children(node, next_player) # curr player corresponds to depth in tree
        self.explored[node] = {"children": children,
                                "unexplored_children": children.copy(),
                                "wins": 0,
                                "visits": 0}
    
    def UCT(self, node):
        """ upper confidence applied to trees formula """
        N = self.explored[node]['visits']
        max_bound = 0
        best_node = None

        for child in self.explored[node]['children']:
            if child not in self.explored:
                continue

            # calculate UCB formula for each child node 
            X = self.explored[child]['wins']
            n = self.explored[child]['visits']
            bound = X + self.exploration_weight * ((np.log(N) / n) ** 0.5)

            if bound > max_bound:
                max_bound = bound # best value seen so far
                best_node = child 

        return best_node 
    
    def update_node(self, node, w):
        """ update node wins and visits """
        self.explored[node]["wins"] += w
        self.explored[node]["visits"] += 1

    def rollout(self, start_state):
        """ run a simulated playout from a given start state"""
        # simulation
        board = UltimateTicTacToeBoard(init_state=start_state)
        player = Player(self.id)
        opponent = Player(self.id % 2 + 1)
        
        done = check_win(board.state)
        result = done
        curr_player = player if len(self.curr_path) % 2 == 1 else opponent
        while not done:
            subgrid, move = player.move(board)
            game_state, result, done = board.subgrid_move(subgrid, curr_player, move)

            # next player
            curr_player = player if curr_player != player else opponent

        win = result == self.id

        # backprop
        for node in self.curr_path:
            self.explored[node]['wins'] += win
            self.explored[node]['visits'] += 1

    def pick_unvisited(self, parent_node):
        """ expand a random child node """
        unvisited = self.explored[parent_node]['unexplored_children']
        selected = np.random.choice(unvisited)
        self.curr_path.append(selected)
        self.add_node(selected)

        # remove child from unexplored list
        self.explored[parent_node]['unexplored_children'].remove(selected)

        return selected

    def run_simulation(self, current):
        self.curr_path.append(current)
        if current not in self.explored:
            self.add_node(current)

        # if current is not a leaf node
        unexplored = self.explored[current]['unexplored_children']
        while len(unexplored) == 0: #TODO : leaf node?
            current = self.UCT(current)

            # check if node is terminal
            if len(self.explored[current]['children']) == 0:
                return current
                
            unexplored = self.explored[current]['unexplored_children']
            self.curr_path.append(current)

        # if node has been rolled out
        if self.explored[current]["visits"] != 0:
            current = self.pick_unvisited(current)

        self.rollout(current)
        self.curr_path = []

    
    def move(self, board):
        state = board.get_str_state()
        for _ in range(10):
            self.run_simulation(state)
        
        move = self.UCT(state)
        idx = int([i for i in range(81) if state[i] != move[i]][0])
        inner_pos = idx % 9
        outer_pos = idx // 9
        return tuple((outer_pos // 3, outer_pos % 3)), tuple((inner_pos // 3, inner_pos % 3))

