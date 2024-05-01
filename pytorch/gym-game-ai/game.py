import time
import gym
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation, FlattenObservation
from wrappers import SkipFrame, ResizeObservation


class Game:
    def __init__(self, render_mode="rgb_array"):
        self.env = gym.make('LunarLander-v2', continuous=False, render_mode=render_mode)
        self.env = FlattenObservation(self.env)
        # env = GrayScaleObservation(env, keep_dim=False)
        # env = TransformObservation(env, f=lambda x: x / 255.)
        self.reset()

    def step_game(self, action):
        state, reward, game_over, _, _ = self.env.step(action)
        return state, reward, game_over

    def get_current_state(self):
        return self.state

    def reset(self):
        self.state, _ = self.env.reset()
        self.reward = 0
        self.game_over = False
        return self.state

    def get_env(self):
        return self.env
