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
            subgrid, move = self.curr_player.move(board)
            game_state, result, done = board.subgrid_move(subgrid, self.curr_player, move)

            # next player
            self.curr_player = self.player1 if self.curr_player != self.player1 else self.player2

        return result