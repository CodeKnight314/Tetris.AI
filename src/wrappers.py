import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict
from typing import Tuple
import cv2
from src.reward_shaper import RewardShaping
from gymnasium import spaces

class TetrisPreprocessor(gym.Wrapper):
    def __init__(self, env, coord: Tuple[int, int, int, int] = (4, 0, 13, 19)):
        super().__init__(env)
        self.coord = coord
        board_space = env.observation_space.spaces["board"]
        assert isinstance(board_space, spaces.Box), "expected board to be a Box"
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.coord[3] - self.coord[1] + 1, self.coord[2] - self.coord[0] + 1),
            dtype=np.uint8
        )

        self.raw_board = None

    def observation(self, obs: dict) -> np.ndarray:
        board = obs["board"].astype(np.uint8)
        binary = np.where(board == 0, 0, 255).astype(np.uint8)
        binary = binary[self.coord[1]:self.coord[3]+1, self.coord[0]:self.coord[2]+1]
        self.raw_board = binary
        return binary

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self.observation(obs), reward, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class ShapedRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_shaper: RewardShaping):
        super().__init__(env)
        self.reward_shaper = reward_shaper

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        raw = None
        wrapper = self.env
        while wrapper is not None:
            if hasattr(wrapper, "raw_board") and wrapper.raw_board is not None:
                raw = wrapper.raw_board
                break
            if hasattr(wrapper, "env"):
                wrapper = wrapper.env
            else:
                wrapper = None

        if raw is None:
            raise RuntimeError("Could not find `raw_board` in any wrapper.")

        shaped_only, reward_dict = self.reward_shaper.calculate_rewards(raw, terminated)

        combined = original_reward + shaped_only

        info["original_reward"] = original_reward
        info["shaped_reward"]   = shaped_only
        info["combined_reward"] = combined
        info["reward_dict"] = reward_dict

        return obs, combined, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max  
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, done, _, info = self.env.step(self.noop_action)
            if done: 
                obs, info = self.env.reset(**kwargs)
        return obs, info
    
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

        obs_space = env.observation_space
        if isinstance(obs_space, Dict):
            obs_shape = obs_space["board"].shape
        elif isinstance(obs_space, Box):
            obs_shape = obs_space.shape
        else:
            raise TypeError("Unsupported observation space type")

        self._obs_buffer = np.zeros((2,) + obs_shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            if isinstance(obs, dict):
                obs_frame = obs["board"]
            else:
                obs_frame = obs

            if i == self._skip - 2: self._obs_buffer[0] = obs_frame
            if i == self._skip - 1: self._obs_buffer[1] = obs_frame

            total_reward += reward
            terminated, truncated = term, trunc
            if terminated or truncated:
                break

        max_frame = self._obs_buffer.max(axis=0)
        if isinstance(obs, dict):
            obs["board"] = max_frame
            return obs, total_reward, terminated, truncated, info
        else:
            return max_frame, total_reward, terminated, truncated, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        assert "FIRE" in action_meanings
        self.fire_action = action_meanings.index("FIRE")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)
