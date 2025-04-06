import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Custom imports
from game import UltimateTicTacToe
from board import UltimateTicTacToeBoard, check_win

# Neural network model for approximating Q-values
class DQN(nn.Module):
    def __init__(self, input_dim=81, output_dim=81):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class QLearningPlayer:
    """
    This class creates a game agent using Deep Q-Learning
    """
    def __init__(self, 
                 learning_rate = 0.001,
                 gamma = 0.99,
                 epsilon = 1.0,
                 epsilon_min = 0.01,
                 epsilon_decay = 0.995,
                 batch_size = 64,
                 target_update_freq = 500,
                 memory_size = 10000,
                 episodes = 5000):
        """ Initialization function for the agent

        Args:
            learning_rate (float): how much the model weights update in response to the error during backpropagation
            gamma (float): how much future rewards are worth compared to immediate rewards
            epsilon (float): starting value for how often the agent explores by picking random actions
            epsilon_min (float): min value that epsilon can decay to
            epsilon_decay (float): rate of how fast epsilon decreases
            batch_size (int): number of samples taken from the replay buffer at each training step
            target_update_freq (int): how often we copy the policy network weights to the target network
            memory_size (int): max size of the replay buffer storing past experiences (state, action, reward, next state, done)
            episodes (int): number of full games the agent will play during training
        """

        # initialize hyperparams
        self.learning_rate = learning_rate,
        self.gamma = gamma,
        self.epsilon = epsilon,
        self.epsilon_min = epsilon_min,
        self.epsilon_decay = epsilon_decay,
        self.batch_size = batch_size,
        self.target_update_freq = target_update_freq,
        self.memory_size = memory_size,
        self.episodes = episodes

        # initialize neural nets
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=self.memory_size)

    def encode_state(self, game):
        board = game.board.get_str_state()  # Should return shape (81,)
        board = [state for state in board]
        return np.array(board, dtype=np.float32)

    # mask Q-values for invalid moves
    # i.e. if your q_vals are [0.1, 0.5, -0.3, 0.9, 0.0], but valid actions are only 1 and 3, 
    # will return masked = [-inf, 0.5, -inf, 0.9, -inf]
    def mask_invalid_actions(self, q_values, valid_actions):
        mask = torch.full_like(q_values, float('-inf'))
        for a in valid_actions:
            mask[a] = q_values[a]
        return mask

    # epsilon-greedy action selection
    def select_action(self, state, valid_actions, epsilon):
        if random.random() < epsilon:
            return random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor).squeeze(0)
            masked_q = self.mask_invalid_actions(q_values, valid_actions)
            return torch.argmax(masked_q).item()

    # optimize model using replay buffer
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, next_valid_batch = zip(*batch)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        # current Q-values
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            max_next_q = []
            for i in range(len(next_valid_batch)):
                masked = self.mask_invalid_actions(next_q_values[i], next_valid_batch[i])
                max_next_q.append(torch.max(masked).item())
            max_next_q = torch.FloatTensor(max_next_q)
            target_q = reward_batch + self.gamma * max_next_q * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def training_loop(self):
        # training loop
        rewards_per_episode = []
        steps_done = 0

        for episode in range(self.episodes):
            game = UltimateTicTacToe()
            state = self.encode_state(game)
            done = False
            total_reward = 0

            while not done:
                valid_actions = game.get_valid_moves()
                if len(valid_actions) == 0:
                    break

                action = self.select_action(state, valid_actions, epsilon)
                row, col = divmod(action, 9)
                game.make_move(row, col)

                reward = 0
                if game.winner == 1:
                    reward = 1
                    done = True
                elif game.winner == -1:
                    reward = -1
                    done = True
                elif game.is_draw():
                    reward = 0.5
                    done = True

                next_state = self.encode_state(game)
                next_valid = game.get_valid_moves()

                self.memory.append((state, action, reward, next_state, done, next_valid))
                state = next_state
                total_reward += reward

                self.optimize_model()

                if steps_done % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                steps_done += 1

            epsilon = max(self.epsilon_min, self.epsilon_decay * epsilon)
            rewards_per_episode.append(total_reward)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")
        

