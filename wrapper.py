import cv2
from collections import deque
import numpy as np
import gym
from random import randint

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

def preprocess(obs):
    obs = np.array(obs)
    if obs.ndim == 3 and obs.shape[2] == 3:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    elif obs.ndim == 2:
        gray = obs
    else:
        raise ValueError(f"Unexpected obs shape: {obs.shape}")
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.deque = deque(maxlen=k)

    def reset(self, raw_obs):
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        frame = preprocess(raw_obs)
        self.deque.clear()
        for _ in range(self.k):
            self.deque.append(frame)
        return np.stack(self.deque, axis=0)

    def step(self, raw_obs):
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]
        frame = preprocess(raw_obs)
        self.deque.append(frame)
        return np.stack(self.deque, axis=0)

def random_init(env,iter,fs):
    for i in range(iter):
        obs,_,_,info = env.step(randint(0,11))
        fs.step(obs)

    return info,obs