import gym
from model import DQNAgent
from collections import deque
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import torch
import cv2
import numpy as np
from image_proccessing import FrameStack

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.env = JoypadSpace(gym_super_mario_bros.make('SuperMarioBros-v0'),COMPLEX_MOVEMENT)
        self.state_stack = deque(maxlen=4)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.agent = DQNAgent(self.state_size,self.action_size)
        self.fs = FrameStack(4)
        self.init = True
        self._first = False

        model_path = "dqn_agent2.pth"
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.agent.target_net.load_state_dict(checkpoint['target_net'])


    def reset(self):
            raw_obs = self.env.reset()[0]
            stacked = self.fs.reset(raw_obs)
            self._first = False
            return stacked


    def preprocess(self, observation):
        # 示例預處理：調整大小、灰度化、歸一化
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        normalized = resized / 255.0
        return normalized

    def act(self, observation):
        processed = self.preprocess(observation)
        # 將當前幀加入緩衝區
        if len(self.state_stack) < 4:
            # 初始填充（假設第一幀重複）
            for _ in range(4):
                self.state_stack.append(processed)
        else:
            self.state_stack.append(processed)
        # 堆疊4幀作為輸入
        state = np.array(self.state_stack)  # Shape: (4, 84, 84)
        state = torch.FloatTensor(state)  # 添加批次維度 -> [1,4,84,84]
        return self.agent.get_action(state, epsilon=0.05)
    

    def step_env(self, action):
        try:
            next_obs, reward, done, info = self.env.step(action)
        except ValueError:
            # env.done 了，直接重設
            next_obs = self.env.reset()[0]
            # 把 FrameStack 也重設
            self.fs.reset(next_obs)
            self._first = False
            # 這一步不算在前一個 episode 裡
            reward, done, info = 0, False, {}
            return next_obs, reward, done, info

        if done:
            self._first = True
        return next_obs, reward, done, info