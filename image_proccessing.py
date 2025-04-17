import cv2
from collections import deque
import numpy as np

def preprocess(obs):
    obs = np.array(obs)
    if obs.ndim == 3 and obs.shape[2] == 3:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    elif obs.ndim == 2:
        gray = obs
    else:
        raise ValueError(f"Unexpected obs shape: {obs.shape}")
    cropped = gray[... , :200]
    resized = cv2.resize(cropped, (84, 84))
    return resized.astype(np.uint8)


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
