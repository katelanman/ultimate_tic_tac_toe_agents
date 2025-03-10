from board import UltimateTicTacToeBoard
from player import Player
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
            curr_subgrid = board.curr_subgrid

            subgrid, move = player.move(board, curr_subgrid)
            
            game_state, result, done = board.subgrid_move(subgrid, player, move)

            # next player
            self.curr_player = self.player1 if player != self.player1 else self.player2

        return result

p1 = Player(1)
p2 = Player(2)

game = UltimateTicTacToe(p1, p2)

results = []
for _ in tqdm(range(10000)):
    results.append(game.play_game())

print(Counter(results))