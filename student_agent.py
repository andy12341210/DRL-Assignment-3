import gym
from model import DQNAgent
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import torch
from image_proccessing import FrameStack

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.env = JoypadSpace(gym_super_mario_bros.make('SuperMarioBros-v0'),COMPLEX_MOVEMENT)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.agent = DQNAgent(self.state_size,self.action_size)
        self.fs = FrameStack(4)
        self.init = True

        model_path = "dqn_agent.pth"
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.agent.target_net.load_state_dict(checkpoint['target_net'])


    def reset(self):
            raw_obs = self.env.reset()[0]
            stacked = self.fs.reset(raw_obs)
            self._first = False
            return stacked


    def act(self, observation):
        if self._first:
            state = self.fs.reset(observation)
            self._first = False
        else:
            state = self.fs.step(observation)

        return self.agent.get_action(state,0)
    
    def step_env(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if done:
            self._first = True
        return next_obs, reward, done, info