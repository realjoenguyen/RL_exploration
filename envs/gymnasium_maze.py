import gymnasium as gym2
import numpy as np
from PIL import Image


class GymnasiumMaze:
    def __init__(self, name, action_repeat=2, size=(64, 64), seed=0):
        self._env = gym2.make(name, render_mode="rgb_array")
        self._action_repeat = action_repeat
        self._size = size

    @property
    def observation_space(self):
        spaces = {}
        spaces["state"] = self._env.observation_space[
            "observation"
        ]  # (4,): (x, y, v_x, v_y)
        spaces["image"] = gym2.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym2.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step, rew, term, trunc, info = self._env.step(action)
            reward += rew
            if term or trunc:
                break
        obs = {}
        obs["state"] = time_step["observation"]
        obs["image"] = self.render()
        obs["is_terminal"] = term
        obs["is_first"] = False  # step fn is always the non-first step
        done = trunc
        return obs, reward, done, info

    def reset(self):
        time_step, _ = self._env.reset()
        obs = {}
        obs["state"] = time_step["observation"]
        obs["image"] = self.render()
        obs["is_terminal"] = False
        obs["is_first"] = True  # reset is always the first time step
        return obs

    def render(self):
        img = self._env.render()
        img = Image.fromarray(img)
        img = img.resize(self._size)
        return np.array(img)
