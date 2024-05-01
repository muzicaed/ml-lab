import time
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from wrappers import SkipFrame, ResizeObservation


# env = gym_super_mario_bros.make('SuperMarioBros-v1')
env = gym.make('CartPole-v0')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Apply Wrappers
# env = GrayScaleObservation(env, keep_dim=False)
# env = ResizeObservation(env, shape=84)
# env = TransformObservation(env, f=lambda x: x / 255.)
# env = SkipFrame(env, skip=4)
# env = FrameStack(env, num_stack=4)

RENDER_SLEEP_TIME = 0.01


class SuperMario:
    def __init__(self):
        self.reset()

    def step_game(self, action):
        state, reward, game_over, _ = env.step(action)
        return state, reward, game_over

    def render_frame(self):
        env.render()
        time.sleep(RENDER_SLEEP_TIME)

    def get_current_state(self):
        return self.state

    def reset(self):
        self.state = env.reset()
        self.reward = 0
        self.game_over = False
        return self.state

    def get_env(self):
        return env


'''
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
'''
