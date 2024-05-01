import os
import torch
import random
import numpy as np
from collections import deque
from snake import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from IPython import display
import matplotlib.pyplot as plt

SAVE_PATH = '/Users/mikaelhellman/dev/ml-lab/model/snake-model.pth'
plt.ion()

MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.003


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.8
        self.memory = deque(maxlen=MAX_MEM)

        self.model = Linear_QNet(11, 500, 250, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food
            game.food.x < game.head.x,  # Left
            game.food.x > game.head.x,  # right
            game.food.y < game.head.y,  # Up
            game.food.y > game.head.y,  # Down
        ]

        return np.array(state, dtype=int)

    def remeber(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # tradeoff explore / eploit
        self.epsilon = 100 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            print('Random move!')
            moveIdx = random.randint(0, 2)
            final_move[moveIdx] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            pred = self.model(state_tensor)
            moveIdx = torch.argmax(pred).item()
            final_move[moveIdx] = 1

        return final_move

    def save(self, high_score):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'high_score': high_score
        }, SAVE_PATH)
        print("Saved model...")

    def load(self):
        if (os.path.exists(SAVE_PATH)):
            checkpoint = torch.load(SAVE_PATH)
            self.model.state_dict(checkpoint['model_state_dict'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.eval()
            self.model.train(True)
            print("Loaded model...")
            return checkpoint['high_score']
        return 0


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    # total_score = agent.load()
    game = SnakeGameAI()

    while True:
        old_state = agent.get_state(game)
        action = agent.get_action(old_state)
        reward, game_over, score = game.play_step(action)
        new_state = agent.get_state(game)
        agent.train_short_memory(old_state, action, reward, new_state, game_over)
        agent.remeber(old_state, action, reward, new_state, game_over)

        if game_over:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.save(total_score)

            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
