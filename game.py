from board import UltimateTicTacToeBoard
from player import Player
from MCTS_player import MCTSPlayer
from tqdm import tqdm

from collections import Counter

class UltimateTicTacToe:
    
    def __init__(self, player1, player2, grid_size=3) -> None:
        self.player1 = player1
        self.player2 = player2
        self.grid_size = grid_size

        self.board = UltimateTicTacToeBoard(self.grid_size)
        self.curr_player = player1

    def play_game(self):
        board = self.board
        board.reset()

        done = False
        while not done:
            player = self.curr_player
            subgrid, move = player.move(board)
            print(subgrid, move)
            print("b state", board.state)
            print("s state", board.subgrids[subgrid].state)
            game_state, result, done = board.subgrid_move(subgrid, player, move)
            print("b state after", board.state)
            print("s state after", board.subgrids[subgrid].state)
            print("empt", board.subgrids[subgrid].num_empty())
            # next player
            self.curr_playe