import gym
from model import DQNAgent
# from RBWmodel import DQNAgent
from collections import deque
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import torch
import cv2
import random
import numpy as np
from wrapper import FrameStack, preprocess

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")



# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.env = JoypadSpace(gym_super_mario_bros.make('SuperMarioBros-v0'),COMPLEX_MOVEMENT)
        # self.env = JoypadSpace(gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0', stages=['1-1']),COMPLEX_MOVEMENT)
        self.state_stack = deque(maxlen=4)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.agent = DQNAgent(self.state_size,self.action_size)
        self.fs = FrameStack(4)
        self.init = True
        self._first = False
        self.a = 0
        self.cnt = 0
        self.n = 0

        model_path = "./final3.pth"
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.agent.target_net.load_state_dict(checkpoint['target_net'])
        self.agent.policy_net.eval()
        set_random_seed(93)


    def reset(self):
            raw_obs = self.env.reset()[0]
            stacked = self.fs.reset(raw_obs)
            self._first = False
            return stacked

    def act(self, observation):
        processed = preprocess(observation)
        if len(self.state_stack) < 4:
            while len(self.state_stack) < 4:
                self.state_stack.append(processed)
        else:
            if self.cnt % 4 == 0:
                self.state_stack.append(processed)
                state = np.array(self.state_stack)
                state = torch.FloatTensor(state)
                self.a = self.agent.get_action(state, epsilon=0.0167)
                self.cnt = 0
                self.n += 1

                if self.n >= 6600:
                    self.a = self.agent.get_action(state, epsilon=0.0167)

        self.cnt += 1
        return self.a
    

    def step_env(self, action):
        try:
            next_obs, reward, done, info = self.env.step(action)
        except ValueError:
            next_obs = self.env.reset()[0]
            self.fs.reset(next_obs)
            self._first = False
            reward, done, info = 0, False, {}
            return next_obs, reward, done, info

        if done:
            self._first = True
        return next_obs, reward, done, info