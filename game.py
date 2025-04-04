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
            game_state, result, done = board.subgrid_move(subgrid, player, move)

            # next player
            self.curr_player = self.player1 if player != self.player1 else self.player2

        return result

p1 = Player(1)
# p2 = Player(2)
p2 = MCTSPlayer(2, exploration_weight=0.01)
p2.train(100000)
# print(p2.explored["0"*82])
game = UltimateTicTacToe(p1, p2)
results = []
for _ in tqdm(range(100)):
    results.append(game.play_ga