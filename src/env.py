import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, ResizeObservation, RecordVideo
from tqdm import tqdm
from src.agent import TetrisAgent
from src.wrappers import MaxAndSkipEnv, TetrisPreprocessor, ShapedRewardWrapper
from src.reward_shaper import RewardShaping
import yaml
import logging
import matplotlib.pyplot as plt
import ale_py

gym.register_envs(ale_py)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TetrisEnv(): 
    def __init__(self, seed: int, num_envs: int, config: str, weights: str = None):
        logger.info(f"Initializing Tetris environment with {num_envs} parallel environments")
        with open(config, 'r') as f: 
            self.config = yaml.safe_load(f)
        
        self.seed = seed
        self.num_envs = num_envs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.set_seed(seed)
        
        self.reward_shaper = RewardShaping(self.config["reward_weights"])
        
        self.env = gym.vector.AsyncVectorEnv(
            [lambda: self._make_env("ALE/Tetris-v5") for i in range(num_envs)], 
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        
        logger.info("Initializing TetrisAgent with configuration:")
        logger.info(f"- Frame stack: {self.config['frame_stack']}")
        logger.info(f"- Learning rate: {self.config['lr']}")
        logger.info(f"- Gamma: {self.config['gamma']}")
        logger.info(f"- Max memory: {self.config['max_memory']}")
        
        self.agent = TetrisAgent(self.config["frame_stack"], 
                                 self.env.action_space[0].n, 
                                 self.config["lr"], 
                                 self.config["gamma"], 
                                 self.config["max_memory"], 
                                 self.config["max_gradient"])
        
        if weights is not None: 
            logger.info(f"Loading pre-trained weights from: {weights}")
            self.agent.load_weights(weights)

        self.history = {
            "reward": [],
            "loss": [],
        }
        
        self.epsilon = self.config["epsilon"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        
        self.max_frames = self.config["max_frames"]
        self.batch_size = self.config["batch_size"]
        self.update_freq = self.config["update_freq"]
        
        self.best_reward = float('-inf')
        self.reward_window_size = 100
        self.save_freq = 1000
        
        logger.info(f"Environment initialized with seed: {seed}")
        
    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def _make_env(self, env_name: str, record_video: bool = False):
        if record_video: 
            env = gym.make(env_name, render_mode="rgb_array")
            video_path = "videos/"
            os.makedirs(video_path)
            env = RecordVideo(env, video_folder=video_path, episode_trigger=lambda x: True)
            env = MaxAndSkipEnv(env, skip=4)
            env = ResizeObservation(env, (84, 84))
            env = FrameStackObservation(env, stack_size=int(self.config["frame_stack"]))
        else:
            env = gym.make(env_name, obs_type="grayscale", render_mode=None)
            env = MaxAndSkipEnv(env, skip=4)
            env = TetrisPreprocessor(env, coord=(27, 203, 22, 63))
            env = ResizeObservation(env, (84, 84))
            env = FrameStackObservation(env, stack_size=4)
            env = ShapedRewardWrapper(env, self.reward_shaper)
        return env

    def train(self, path: str):
        logger.info(f"Starting training process. Model will be saved to: {path}")
        os.makedirs(path, exist_ok=True)

        total_frames = 0

        episode_rewards = np.zeros(self.num_envs, dtype=float)

        state, _ = self.env.reset()  

        pbar = tqdm(total=self.max_frames, desc="Frames")

        while total_frames < self.max_frames:
            epsilon = max(self.epsilon_min, self.epsilon - (total_frames * self.epsilon_decay / self.max_frames))
            actions = []
            for i in range(self.num_envs):
                single_board = state[i]
                a_i = self.agent.select_action(single_board, epsilon)
                actions.append(int(a_i))
            actions = np.array(actions, dtype=np.int32)

            next_state, rewards, terminateds, truncateds, infos = self.env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(self.num_envs):
                prev_board = torch.tensor(state[i]).float()
                next_board = torch.tensor(next_state[i]).float()
                r_i = float(rewards[i])
                d_i = bool(dones[i])

                self.agent.push(prev_board, actions[i], r_i, next_board, d_i)
                episode_rewards[i] += r_i

                if d_i:
                    self.history["reward"].append(float(episode_rewards[i]))
                    episode_rewards[i] = 0.0

            if len(self.agent.buffer) > self.batch_size:
                loss = self.agent.update(self.batch_size)
                self.history["loss"].append(loss)

            total_frames += self.num_envs
            pbar.update(self.num_envs)

            if total_frames % self.update_freq == 0:
                self.agent.update_target_network(hard_update=True)
                logger.info(f"Target network updated at frame {total_frames}")

            if len(self.history["reward"]) >= self.reward_window_size:
                recent_reward_avg = np.mean(self.history["reward"][-self.reward_window_size:])
                if recent_reward_avg > self.best_reward:
                    self.best_reward = recent_reward_avg
                    best_model_path = os.path.join(path, "best_model.pth")
                    self.agent.save_weights(best_model_path)
                    logger.info(f"New best model saved! Average reward: {recent_reward_avg:.2f}")

            if total_frames % self.save_freq == 0:
                checkpoint_path = os.path.join(path, f"checkpoint_{total_frames}.pth")
                self.agent.save_weights(checkpoint_path)
                logger.info(f"Checkpoint saved at frame {total_frames}")

            state = next_state

            pbar.set_postfix(reward=np.mean(self.history["reward"][-10:]), loss=np.mean(self.history["loss"][-10:]))

        pbar.close()
        logger.info("Training completed. Saving final model weights...")
        self.agent.save_weights(os.path.join(path, "final_model.pth"))
        logger.info(f"Final model weights saved to: {os.path.join(path, 'final_model.pth')}")

    def test(self, output_path: str, num_episodes: int):
        os.makedirs(output_path, exist_ok=True)
        
        self.env = self._make_env("tetris_gymnasium/Tetris", True)
        episodes_reward = []
        episodes_length = []
        
        self.agent.model.eval()
        
        logger.info(f"Starting testing phase with {num_episodes} episodes")
        for i in range(num_episodes):
            logger.info(f"Recording episode {i + 1}/{num_episodes}")
            
            state, info = self.env.reset()
            episode_reward = 0
            episdoe_length = 0
            done = False
            
            while not done: 
                if isinstance(state, dict):
                    board = state["board"]
                else: 
                    board = state
                    
                with torch.no_grad():
                    action = self.agent.select_action(action, epsilon=0.0)
                    
                next_state, reward, terminated, truncated, _ = self.env.step(action)
        
    
    def close(self):
        self.env.close() 
        del self.agent
        torch.cuda.empty_cache()
    
    def save_weights(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.agent.save_weights(path)

    def plot_history(self, path: str):
        plt.figure(figsize=(12, 6))
        plt.plot(self.history["reward"], label="Reward")
        plt.plot(self.history["loss"], label="Loss")
        plt.legend()
        plt.savefig(os.path.join(path, "history.png"))
        plt.close()