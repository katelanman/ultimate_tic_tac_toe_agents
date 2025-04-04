import numpy as np
import time

"""
tic tac toe board. 
0 - empty square
1 - X's
2 - O's
"""
def check_win(grid) -> int:
    """ Check whether there is a winner """
    # check row/col win
    for i in range(len(grid)):
        row = list(set(grid[i,:]))
        col = list(set(grid[:,i]))
        if len(row) == 1 and row[0] != 0:
            return row[0]
        if len(col) == 1 and col[0] != 0:
            return col[0]
        
    # check diagonal wins
    diag = list(set(grid[i, i] for i in range(len(grid))))
    back_diag = list(set(grid[i, len(grid) - 1 - i] for i in range(len(grid))))
    if len(diag) == 1 and diag[0] != 0:
        return diag[0]
    if len(back_diag) == 1 and back_diag[0] != 0:
        return back_diag[0]
    
    return 0

class Board:
    def __init__(self, grid_size=3, init_state=None) -> None:
        self.grid_size = grid_size
        self.state = np.zeros((self.grid_size, self.grid_size))
        self.playable = True

        if init_state:
            self._update_board(init_state)

    def _update_board(self, state):
        self.state = np.array([np.array(list(state[(i)*self.grid_size:(i+1)*self.grid_size])).astype(int) 
                               for i in range(0, self.grid_size)])
        if check_win(self.state):
            self.playable = False

    def reset(self):
        """ reset board to be empty """
        self.state = np.zeros((self.grid_size, self.grid_size))

    def move(self, player, pos) -> tuple[np.array, int, bool]:
        """ 
        player moves in square pos=(x, y) 
        Returns:
            game state, game result (winning player id, -1 for tie, 0 for still going), whether the game is done
        """
        if self.state[pos] != 0:
            print('Illegal move')
            raise ValueError("Invalid move")
        
        self.state[pos] = player.id

        if self.check_win(player, pos):
            self.playable = False
            return self.state, player.id, True

        if self.num_empty() == 0:
            self.playable = False
            return self.state, -1, True

        return self.state, 0, False
    
    def check_win(self, last_player, last_pos) -> bool:
        """ Check whether either side has won the game """
        # check row win
        for i in range(self.grid_size):
            if self.state[last_pos[0],i] != last_player.id:
                break
            if i == self.grid_size - 1:
                return True

        # check column win
        for i in range(self.grid_size):
            if self.state[i,last_pos[1]] != last_player.id:
                break
            if i == self.grid_size - 1:
                return True
            
        # check diagonal wins
        for i in range(self.grid_size):
            if self.state[i,i] != last_player.id:
                break
            if i == self.grid_size - 1:
                return True
            
        for i in range(self.grid_size):
            if self.state[i,self.grid_size - 1 - i] != last_player.id:
                break
            if i == self.grid_size - 1:
                return True

        return False
    
    def num_empty(self):
        return (self.grid_size ** 2) - np.count_nonzero(self.state)

class UltimateTicTacToeBoard(Board):
    def __init__(self, grid_size=3, init_state=None) -> None:
        super(UltimateTicTacToeBoard, self).__init__(grid_size)
        self.subgrids = np.array([[Board(self.grid_size) for _ in range(self.grid_size)] for _ in range(self.grid_size)])

        self.curr_subgrid = None
        if init_state:
            self._update_board(init_state)

    def _update_board(self, state):
        """ 
        set board configuration given a string representation of the state 
        params:
            state: first (n*n)^2 characters indicate the status of each square (0 - empty, 1,2 - player 1 or 2),
                   final characters represent the current playable subgrid (0 - all, 1,...,(n*n))
        """
        n = self.grid_size
        self.subgrids = np.array(
            [[Board(n, state[(i)*(n**3):(i+1)*(n**3)][j*(n**2):(j+1)*(n**2)]) for j in range(n)] for i in range(n)]
        )
        self.state = np.array([[check_win(self.subgrids[i,j].state) for j in range(n)] for i in range(n)])

        self.curr_subgrid = tuple([(int(state[n**4:]) - 1) // 3, (int(state[n**4:]) - 1) % 3])
        if not self.subgrids[self.curr_subgrid].playable:
            self.curr_subgrid = None

    def get_str_state(self):
        """ 
        gets state as a len (n*n)^2 + x str representation; 
        first (n*n)^2 characters indicate the status of each square (0 - empty, 1,2 - player 1 or 2)
        final characters represent the current playable subgrid (0 - all, 1,...,(n*n))
        """
        states = [str(i) for b in self.subgrids.flatten() for i in b.state.astype(int).flatten()]
        curr = [str(self.grid_size*self.curr_subgrid[0] + self.curr_subgrid[1] + 1) if self.curr_subgrid else "0"]
        
        return ''.join(states + curr)

    def subgrid_move(self, subgrid_pos, player, pos) -> tuple[np.array, int, bool]:
        """ 
        player moves in square pos=(x, y) of subgrid=(X, Y)
        Returns:
            game state, game result (winning player id, -1 for tie, 0 for still going), whether the game is done
        """
        subgrid = self.subgrids[subgrid_pos]
        next_subgrid = self.subgrids[pos]

        sub_state, result, done = subgrid.move(player, pos)

        # no more playable squares in next subgrid
        if not next_subgrid.playable or next_subgrid.num_empty == 0:
            self.curr_subgrid = None 
        else:
            self.curr_subgrid = pos 
        
        if done:
            self.state[subgrid_pos] = result

        if self.check_win(player, subgrid_pos):
            return self.state, player.id, True

        if self.num_empty() == 0:
            return self.state, -1, True

        return self.state, 0, False

    def check_win(self, last_player, last_pos) -> bool:
        return super().check_win(last_player, last_pos)
    
    def num_empty(self):
        return super().num_empty()

    def reset(self):
        super().reset()

        # reset all sub-grids
        for grid in self.subgrids.flatten():
            grid.reset()
            grid.playable = True

        self.curr_subgrid = None
