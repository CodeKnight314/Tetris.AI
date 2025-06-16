import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, ResizeObservation
from tqdm import tqdm
from src.agent import TetrisAgent
from src.wrappers import TetrisPreprocessor, ShapedRewardWrapper
from src.reward_shaper import RewardShaping
import yaml
import logging
import matplotlib.pyplot as plt
from tetris_gymnasium.envs import Tetris
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TetrisEnv(): 
    def __init__(self, seed: int, num_envs: int, config: str, weights: str = None, verbose: bool = True):
        logger.info(f"Initializing Tetris environment with {num_envs} parallel environments")
        with open(config, 'r') as f: 
            self.config = yaml.safe_load(f)
        
        self.seed = seed
        self.num_envs = num_envs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.buffer_type = self.config["buffer_type"]
        logger.info(f"Using device: {self.device}")

        self.set_seed(seed)
        
        self.reward_shaper = RewardShaping(self.config["reward_weights"])
        
        self.env = gym.vector.AsyncVectorEnv(
            [lambda: self._make_env() for i in range(num_envs)], 
            autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
        )
        
        logger.info("Initializing TetrisAgent with configuration:")
        logger.info(f"- Frame stack: {self.config['frame_stack']}")
        logger.info(f"- Learning rate: {self.config['lr']}")
        logger.info(f"- Gamma: {self.config['gamma']}")
        logger.info(f"- Max memory: {self.config['max_memory']}")
        logger.info(f"- Weights: {self.config['reward_weights']}")
        logger.info(f"- Action Space: {self.env.action_space[0].n}")
        
        self.agent = TetrisAgent(self.config["frame_stack"], 
                                 self.env.action_space[0].n, 
                                 self.config["lr"], 
                                 self.config["gamma"], 
                                 self.config["max_memory"], 
                                 self.config["max_gradient"], 
                                 self.config["action_mask"], 
                                 self.buffer_type)
        
        if weights is not None: 
            logger.info(f"Loading pre-trained weights from: {weights}")
            self.agent.load_weights(weights)

        self.history = {
            "reward": deque(maxlen=self.config["window_size"] * 2),
            "loss": deque(maxlen=self.config["window_size"] * 2),
        }
        
        self.epsilon = self.config["epsilon"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        
        self.max_frames = self.config["max_frames"]
        self.batch_size = self.config["batch_size"]
        self.update_freq = self.config["update_freq"]
        self.reward_window_size = self.config["window_size"]
        
        self.best_reward = float('-inf')
        self.save_freq = self.config["save_freq"]
        self.verbose = verbose
        
        logger.info(f"Environment initialized with seed: {seed}")
        
    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def _make_env(self, render_mode: str = None):
        env = gym.make("tetris_gymnasium/Tetris", render_mode=render_mode)
        env = TetrisPreprocessor(env)
        env = ResizeObservation(env, (84, 84))
        env = ShapedRewardWrapper(env, self.reward_shaper)
        env = FrameStackObservation(env, stack_size=int(self.config["frame_stack"]))
        return env

    def train(self, path: str):
        logger.info(f"Starting training process. Model will be saved to: {path}")
        os.makedirs(path, exist_ok=True)

        total_frames = 0

        episode_rewards = np.zeros(self.num_envs, dtype=float)

        state, _ = self.env.reset()  

        pbar = tqdm(total=self.max_frames, desc="Frames")

        while total_frames < self.max_frames:
            epsilon = max(self.epsilon_min, self.epsilon * (self.epsilon_decay ** (total_frames / 1000)))

            actions = []
            q_values = []
            for i in range(self.num_envs):
                single_board = state[i]
                output = self.agent.select_action(single_board, epsilon)
                if isinstance(output, tuple):
                    a_i, q_value = output
                    actions.append(int(a_i))
                    q_values.append(q_value)
                else: 
                    a_i = output
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

                total_frames += 1

                if total_frames % self.update_freq == 0:
                    self.agent.update_target_network(hard_update=True)
                    logger.info(f"Target network updated at frame {total_frames}")

                if total_frames % self.save_freq == 0:
                    checkpoint_path = os.path.join(path, f"checkpoint.pth")
                    self.agent.save_weights(checkpoint_path)
                    if self.verbose:
                        logger.info(f"Checkpoint saved at frame {total_frames}")

            pbar.update(self.num_envs)

            if len(self.agent.buffer) > self.batch_size:
                loss = self.agent.update(self.batch_size)
                self.history["loss"].append(loss)

            if len(self.history["reward"]) >= self.reward_window_size:
                recent_reward_avg = np.mean(self.history["reward"])
                if recent_reward_avg > self.best_reward:
                    self.best_reward = recent_reward_avg
                    best_model_path = os.path.join(path, "best_model.pth")
                    self.agent.save_weights(best_model_path)
                    self.test(os.path.join(path, "video"), num_episodes=1)
                    logger.info(f"New best model saved! Average reward: {recent_reward_avg:.2f}")

                    all_rewards_dict = {}
                    for key, value in infos["reward_dict"].items():
                        if key.startswith("_"):
                            continue
                        if key not in all_rewards_dict:
                            all_rewards_dict[key] = []
                        all_rewards_dict[key].extend(value if isinstance(value, list) else [value])
                    
                    for key, values in all_rewards_dict.items():
                        logger.info(f"> {key}: {np.mean(values):.4f}")

            state = next_state

            q_val_mean = np.mean(q_values) if len(q_values) > 0 else 0.0
            q_val_std = np.std(q_values) if len(q_values) > 0 else 0.0
            pbar_rewards = np.mean(self.history['reward']) if len(self.history["reward"]) > 0 else 0.0
            pbar_loss = np.mean(self.history['loss']) if len(self.history["reward"]) > 0 else 0.0
            pbar.set_postfix(
                reward=f"{pbar_rewards:.4f}", 
                loss=f"{pbar_loss:.4f}", 
                epsilon=f"{epsilon:.4f}",
                beta=f"{self.agent.beta:.4f}",
                q_values=f"{q_val_mean:.4f}",
                q_values_std=f"{q_val_std:.4f}"
            )

        pbar.close()
        logger.info("Training completed. Saving final model weights...")
        self.agent.save_weights(os.path.join(path, "final_model.pth"))
        logger.info(f"Final model weights saved to: {os.path.join(path, 'final_model.pth')}")

    def test(self, path: str, num_episodes: int):
        import cv2
        os.makedirs(path, exist_ok=True)
        
        env = self._make_env(render_mode="rgb_array")
        self.agent.model.eval()

        state, _ = env.reset()
        frame = env.render()
        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(path, "tetris.mp4")
        video = cv2.VideoWriter(video_path, fourcc, 60, (width, height))

        total_rewards = 0
        total_steps = 0

        for i in range(num_episodes):
            state, _ = env.reset()
            done = False
            rewards = 0
            steps = 0
            while not done:
                frame = env.render()
                video.write(frame)
                action, _ = self.agent.select_action(state, 0.0)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                rewards += reward
                steps += 1

            if self.verbose:
                logger.info(f"Episode {i + 1} - Reward: {rewards:.2f} - Steps: {steps}")
            total_rewards += rewards    
            total_steps += steps

        avg_reward = total_rewards / num_episodes
        avg_steps = total_steps / num_episodes

        if self.verbose:
            logger.info(f"Average reward: {avg_reward:.2f} - Average steps: {avg_steps:.2f}")

        video.release()
        if self.verbose:
            logger.info(f"Video saved to: {video_path}")
        del env

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