import numpy as np
from agent import Agent
from plot import plot_learning_curve
from game import Game

NO_OF_GAMES = 300
STEPS_BEFORE_LEARN = 20
PLOT_FILE = '/Users/mikaelhellman/dev/ml-lab/pytorch/gym-game-ai/plot_learning.png'
NO_OF_EPOCHS = 4

if __name__ == '__main__':
    game = Game()
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

    # agent.load_models()

    for i in range(NO_OF_GAMES):
        state = game.reset()
        done = False
        reward_score = 0

        while not done:
            steps_count += 1
            action, probs, value = agent.choose_action(state)
            new_state, reward, done = game.step_game(action)
            agent.remember(state, action, probs, value, reward, done)
            reward_score += reward

            if (steps_count % STEPS_BEFORE_LEARN == 0):
                agent.learn()
                learn_iterations += 1

            state = new_state
            game.render_frame()

        reward_score_history.append(reward_score)
        avg_reward_score = np.mean(reward_score_history[-100:])

        if avg_reward_score > best_score:
            best_score = avg_reward_score
            agent.save_models()

        print(f'Game: {i}, Score: {reward_score}, Avg: {int(avg_reward_score)}, Steps: {steps_count}, Learn itr: {learn_iterations}')

    x = [i + 1 for i in range(len(reward_score_history))]
    plot_learning_curve(x, reward_score_history, PLOT_FILE)
