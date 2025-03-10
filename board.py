import numpy as np

"""
tic tac toe board. 
0 - empty square
1 - O's
2 - X's
"""
class Board:
    def __init__(self, grid_size=3) -> None:
        self.grid_size = grid_size
        self.state = np.zeros((self.grid_size, self.grid_size))
        self.playable = True

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
    def __init__(self, grid_size=3) -> None:
        super(UltimateTicTacToeBoard, self).__init__(grid_size)
        self.subgrids = np.array([[Board(self.grid_size) for _ in range(self.grid_size)] for _ in range(self.grid_size)])

        self.curr_subgrid = None

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

    

    