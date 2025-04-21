from player import Player
from tabularQL_player import TabularQPlayer
from game import UltimateTicTacToe
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

def train_against_random(episodes=5000, eval_interval=500):
    """train q-learning agent against random player and evaluate performance"""
    q_agent = TabularQPlayer(id=1, alpha=0.1, gamma=0.9, epsilon=1.0, save_path="q_table_vs_random.pkl")
    random_player = Player(id=2)
    
    win_rates = []
    tie_rates = []
    loss_rates = []
    episodes_x = []
    
    for i in range(0, episodes, eval_interval):
        # train for eval_interval episodes
        q_agent.train(random_player, episodes=eval_interval, show_progress=True)
        
        # evaluate performance
        wins, ties, losses = evaluate_agent(q_agent, random_player, eval_episodes=100)
        win_rate = wins / 100
        tie_rate = ties / 100
        loss_rate = losses / 100
        
        win_rates.append(win_rate)
        tie_rates.append(tie_rate)
        loss_rates.append(loss_rate)
        episodes_x.append(i + eval_interval)
        
        print(f"After {i + eval_interval} episodes:")
        print(f"Win rate: {win_rate:.2f}, Tie rate: {tie_rate:.2f}, Loss rate: {loss_rate:.2f}")
        
    # plot results
    plt.figure(figsize=(10, 6))
    plt.plot(episodes_x, win_rates, label='Win Rate')
    plt.plot(episodes_x, tie_rates, label='Tie Rate')
    plt.plot(episodes_x, loss_rates, label='Loss Rate')
    plt.xlabel('Training Episodes')
    plt.ylabel('Rate')
    plt.title('Q-Learning Agent Performance vs Random Player')
    plt.legend()
    plt.grid(True)
    plt.savefig('q_learning_performance.png')
    plt.show()
    
    return q_agent

if __name__ == "__main__":
    # uncomment to run
    q_agent = train_against_random(episodes=100000)
    
    random_player = Player(id=2)
    game = UltimateTicTacToe(q_agent, random_player)
    result = game.play_game()
    print(f"Game result: {result}")