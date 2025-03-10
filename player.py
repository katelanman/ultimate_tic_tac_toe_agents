import random
import numpy as np

class Player:
    def __init__(self, id) -> None:
        self.id = id

    def move(self, board, subgrid) -> tuple[int, int]:
        # TODO: some function to determine players next move

        # randomly choose an empty square
        if subgrid is None: # can move anywhere
            subgrid = tuple(random.choice(np.array(np.where(board.state == 0)).T))

        pos = tuple(random.choice(np.array(np.where(board.subgrids[subgrid].state == 0)).T))
        return subgrid, pos 