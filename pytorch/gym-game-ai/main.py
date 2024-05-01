import gym
import numpy as np
from agent import Agent
from plot import plot_learning_curve
from game import Game

NO_OF_GAMES = 500
STEPS_BEFORE_LEARN = 50
PLOT_FILE = '/Users/mikaelhellman/dev/ml-lab/pytorch/gym-game-ai/plot_learning.png'
NO_OF_EPOCHS = 4
ALLOWED_STEPS_PER_GAME = 2000

if __name__ == '__main__':
    game = Game("human")
    env = game.get_env()

    best_score = env.reward_range[0]
    reward_score_history = []
    learn_iterations = 0
    avg_reward_score = 0
    steps_count = 0

    agent = Agent(actions_count=env.action_space.n,
                  inputs_dim=env.observation_space.shape,
                  learning_rate=0.0003,
                  discount_factor=0.99,
                  tradeoff=0.95,
                  no_of_epochs=NO_OF_EPOCHS,
                  batch_size=5)

    agent.load_models()

    for i in range(NO_OF_GAMES):
        state = game.reset()
        done = False
        reward_score = 0
        steps_in_game_count = 0

        while not done:
            steps_count += 1
            steps_in_game_count += 1
            action, probs, value = agent.choose_action(state)
            new_state, reward, done = game.step_game(action)
            agent.remember(state, action, probs, value, reward, done)
            reward_score += reward
            state = new_state

            if (steps_count % STEPS_BEFORE_LEARN == 0):
                agent.learn()
                learn_iterations += 1

            if steps_in_game_count > ALLOWED_STEPS_PER_GAME:
                break

        reward_score_history.append(reward_score)
        avg_reward_score = np.mean(reward_score_history[-50:])

        if avg_reward_score > best_score or reward_score * 3 > best_score:
            best_score = avg_reward_score
            agent.save_models()

        print(f'Game: {i}, Score: {int(reward_score)}, Avg: {int(avg_reward_score)}, Steps: {steps_in_game_count}, Learn itr: {learn_iterations}')

    x = [i + 1 for i in range(len(reward_score_history))]
    plot_learning_curve(x, reward_score_history, PLOT_FILE)
