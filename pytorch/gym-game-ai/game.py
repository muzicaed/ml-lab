import time
import gym
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from wrappers import SkipFrame, ResizeObservation

env = gym.make('CartPole-v0', render_mode="human")

# Apply Wrappers
# env = GrayScaleObservation(env, keep_dim=False)
# env = ResizeObservation(env, shape=84)
# env = TransformObservation(env, f=lambda x: x / 255.)
# env = SkipFrame(env, skip=4)
# env = FrameStack(env, num_stack=4)

RENDER_SLEEP_TIME = 0.01


class Game:
    def __init__(self):
        self.reset()

    def step_game(self, action):
        state, reward, game_over, _, _ = env.step(action)
        return state, reward, game_over

    def render_frame(self):
        env.render()
        # time.sleep(RENDER_SLEEP_TIME)

    def get_current_state(self):
        return self.state

    def reset(self):
        self.state, _ = env.reset()
        self.reward = 0
        self.game_over = False
        return self.state

    def get_env(self):
        return env
