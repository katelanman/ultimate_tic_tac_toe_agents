import random
import numpy as np

class Player:
    def __init__(self, id) -> None:
        self.id = id

    def move(self, board) -> tuple[int, int]:
        """
        random agent for playing ultimate tic tac toe
        Params:
            board (Object) - ultimate tic tac toe game object
            subgrid (tuple) - tuple indicating the current subgrid to play in
        """
        # TODO: some function to determine players next move
        subgrid = board.curr_subgrid

        # randomly choose an empty square
        if subgrid is None: # can move anywhere
            print("subgrid is None")
            subgrid = tuple(random.choice(np.array(np.where(board.state == 0)).T))
            print("subgrid:", subgrid)
            test = len(np.array(np.where(board.subgrids[subgrid].state == 0)).T) == 0
            if test: 
                print("test true:", subgrid)
                print(board.subgrids[subgrid].state)

        pos = tuple(random.choice(np.array(np.where(board.subgrids[subgrid].state == 0)).T))
        return subgrid, pos 