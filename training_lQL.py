from board import UltimateTicTacToeBoard
from player import Player
from linearQL_player import LinearQPlayer
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os

class QTrainer:
    def __init__(self, episodes=5000, stats_interval=500, save_dir="training_stats"):
        """
        initializes new object for class

        params:
            episodes: num episodes to train on
            stats_interval: how many eps to print stats after
            save_dir: file path to save stats to
        """
        self.episodes = episodes
        self.stats_interval = stats_interval  
        self.save_dir = save_dir
        
        # create directory to save stats to if it doesn't exist 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # initialize 
        self.win_rates = []
        self.loss_rates = []
        self.draw_rates = []
        
    def train_vs_random(self, q_player):
        """train q-learning agent against a random player"""
        random_player = Player(3 - q_player.id)  # opponent has the other id
        
        print("Training Q-learning agent against random player...")
        wins, losses, draws = 0, 0, 0
        
        for episode in tqdm(range(self.episodes)):
            board = UltimateTicTacToeBoard()
            current_player = q_player if episode % 2 == 0 else random_player  # alternate who goes first
            
            done = False
            move_count = 0
            
            while not done:
                subgrid, pos = current_player.move(board)
                
                # get reward (-0.01 per move to encourage faster wins)
                reward = -0.01
                
                # make move
                game_state, result, done = board.subgrid_move(subgrid, current_player, pos)
                
                # update rewards based on game result
                if done:
                    # we won
                    if result == q_player.id:  
                        reward = 1.0
                        wins += 1
                    # tie
                    elif result == -1: 
                        reward = 0.2
                        draws += 1
                    # we lost
                    else:  
                        reward = -1.0
                        losses += 1
                
                # update q-values
                if current_player.id == q_player.id:
                    q_player.post_move_update(board, reward, done)
                
                # switch players
                current_player = random_player if current_player.id == q_player.id else q_player
                move_count += 1
            
            # collect statistics at interval
            if (episode + 1) % self.stats_interval == 0:

                total = wins + losses + draws
                self.win_rates.append(wins / total)
                self.loss_rates.append(losses / total)
                self.draw_rates.append(draws / total)
                
                # print progress 
                print(f"Episode {episode+1}/{self.episodes}: Win Rate: {wins/total:.2f}, Draw Rate: {draws/total:.2f}, Loss Rate: {losses/total:.2f}")
                
                # reset counters
                wins, losses, draws = 0, 0, 0
    
    def train_vs_self(self, q_player1, q_player2):
        """train two q-learning agents against each other"""
        print("Training Q-learning agents against each other...")
        wins1, wins2, draws = 0, 0, 0
        
        for episode in tqdm(range(self.episodes)):
            board = UltimateTicTacToeBoard()
            first_player = q_player1 if episode % 2 == 0 else q_player2  # alternate who goes first
            second_player = q_player2 if episode % 2 == 0 else q_player1
            current_player = first_player
            
            done = False
            move_count = 0
            
            while not done:
                # get current player's move
                subgrid, pos = current_player.move(board)
                
                # get reward
                reward = -0.01  # small penalty for each move to encourage faster wins
                
                # make the move
                game_state, result, done = board.subgrid_move(subgrid, current_player, pos)
                
                # update rewards based on game result
                if done:
                    if result == q_player1.id:  # player 1 won
                        wins1 += 1
                        q_player1.post_move_update(board, 1.0, done)
                        q_player2.post_move_update(board, -1.0, done)
                    elif result == q_player2.id:  # player 2 won
                        wins2 += 1
                        q_player1.post_move_update(board, -1.0, done)
                        q_player2.post_move_update(board, 1.0, done)
                    else:  # draw
                        draws += 1
                        q_player1.post_move_update(board, 0.2, done)
                        q_player2.post_move_update(board, 0.2, done)
                else:
                    # update q-values for current player
                    if current_player.id == q_player1.id:
                        q_player1.post_move_update(board, reward, done)
                    else:
                        q_player2.post_move_update(board, reward, done)
                
                # switch players
                current_player = second_player if current_player.id == first_player.id else first_player
                move_count += 1
            
            # collect statistics at intervals but don't save the models
            if (episode + 1) % self.stats_interval == 0:
                # calculate and store statistics
                total = wins1 + wins2 + draws
                self.win_rates.append(wins1 / total)
                self.loss_rates.append(wins2 / total)
                self.draw_rates.append(draws / total)
                
                # print progress update
                print(f"Episode {episode+1}/{self.episodes}: P1 Win Rate: {wins1/total:.2f}, Draw Rate: {draws/total:.2f}, P2 Win Rate: {wins2/total:.2f}")
                
                # reset counters
                wins1, wins2, draws = 0, 0, 0
    
    def evaluate(self, q_player, opponent, num_games=100):
        """evaluate a trained q-player against an opponent"""
        # turn off training mode
        q_player.is_training = False  
        
        wins, losses, draws = 0, 0, 0
        
        print(f"Evaluating Q-learning agent against {opponent.__class__.__name__}...")
        for game in tqdm(range(num_games)):
            board = UltimateTicTacToeBoard()
            # alternate who goes first
            first_player = q_player if game % 2 == 0 else opponent  
            second_player = opponent if game % 2 == 0 else q_player
            current_player = first_player
            
            done = False
            
            while not done:
                # get current player's move
                subgrid, pos = current_player.move(board)
                
                # make the move
                game_state, result, done = board.subgrid_move(subgrid, current_player, pos)
                
                # record result
                if done:
                    # we won
                    if result == q_player.id:  
                        wins += 1
                    # tie
                    elif result == -1:  
                        draws += 1
                    # we lost
                    else:  
                        losses += 1
                
                # switch players
                current_player = second_player if current_player.id == first_player.id else first_player
        
        win_rate = wins / num_games
        draw_rate = draws / num_games
        loss_rate = losses / num_games
        
        print(f"Results: Win rate: {win_rate:.2f}, Draw rate: {draw_rate:.2f}, Loss rate: {loss_rate:.2f}")
        return win_rate, draw_rate, loss_rate
    
    def plot_performance(self):
        """plot the agent's performance over training"""
        episodes = [i * self.stats_interval for i in range(len(self.win_rates))]
        
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, self.win_rates, 'g-', label='Win Rate')
        plt.plot(episodes, self.draw_rates, 'b-', label='Draw Rate')
        plt.plot(episodes, self.loss_rates, 'r-', label='Loss Rate')
        plt.xlabel('Episodes')
        plt.ylabel('Rate')
        plt.title('Agent Performance During Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/training_performance.png")
        plt.show()
        
        # save stats to file
        stats = {
            'episodes': episodes,
            'win_rates': self.win_rates,
            'draw_rates': self.draw_rates,
            'loss_rates': self.loss_rates
        }
        with open(f"{self.save_dir}/training_stats.pkl", 'wb') as f:
            pickle.dump(stats, f)


if __name__ == "__main__":

    q_player = LinearQPlayer(id=1, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.999)
    
    # create trainer
    trainer = QTrainer(episodes=5000, stats_interval=500)
    
    # train 
    trainer.train_vs_random(q_player)
    
    # plot training performance
    trainer.plot_performance()
    
    # evaluate 
    random_opponent = Player(id=2)
    trainer.evaluate(q_player, random_opponent, num_games=100)
    
   