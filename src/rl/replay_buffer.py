from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.buf = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def push(self, obs, action, reward, next_obs, done):
        # store as small numpy arrays for speed
        self.buf.append((
            np.asarray(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, act, rew, nxt, done = zip(*batch)

        return (
            np.stack(obs),
            np.array(act, dtype=np.int64),
            np.array(rew, dtype=np.float32),
            np.stack(nxt),
            np.array(done, dtype=np.float32),
        )