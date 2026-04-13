import os
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from tetris_gymnasium.envs import Tetris
from src.agent import LinearAgent
from src.wrappers import PlacementEnv
from src.placement import (
    enumerate_placements, enumerate_placements_with_hold,
    NUM_FEATURES,
)
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def play_game(weights, use_hold=True, lookahead=1, top_k=5, max_pieces=50000):
    """Play one complete game and return lines cleared.

    Top-level function for multiprocessing compatibility.
    """
    agent = LinearAgent(weights)
    env = PlacementEnv(Tetris(gravity=False))
    env.reset()
    board = env.get_board_binary()

    total_lines = 0
    pieces = 0

    while pieces < max_pieces:
        piece_id = env.get_piece_id()

        if use_hold:
            hold_id = env.get_hold_piece_id()
            next_id = env.get_next_piece_id()
            has_swapped = env.get_has_swapped()
            placements = enumerate_placements_with_hold(
                board, piece_id, hold_id, next_id, has_swapped
            )
        else:
            placements = enumerate_placements(board, piece_id)

        if not placements:
            break

        next_piece_id = env.get_next_piece_id() if lookahead >= 2 else None
        idx = agent.select_placement(placements, next_piece_id, lookahead, top_k)
        if idx is None:
            break

        placement = placements[idx]
        _, _, terminated, truncated, _ = env.execute_placement(placement)

        total_lines += placement.lines_cleared
        pieces += 1

        if terminated or truncated:
            break

        board = env.get_board_binary()

    env.close()
    return total_lines


def evaluate_weights(args):
    """Evaluate a weight vector by playing num_games games. For multiprocessing."""
    weights, num_games, use_hold, lookahead, top_k = args
    total = sum(play_game(weights, use_hold, lookahead, top_k)
                for _ in range(num_games))
    return total / num_games


class CEMTrainer:
    def __init__(self, config: str, verbose: bool = True):
        logger.info("Initializing CEM Trainer")
        with open(config, 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_features = self.config.get("num_features", NUM_FEATURES)
        self.population_size = self.config.get("population_size", 100)
        self.elite_frac = self.config.get("elite_frac", 0.1)
        self.elite_count = max(1, int(self.population_size * self.elite_frac))
        self.num_games = self.config.get("num_games", 5)
        self.num_generations = self.config.get("num_generations", 200)
        self.noise_initial = self.config.get("noise_initial", 5.0)
        self.noise_decay = self.config.get("noise_decay", 0.1)
        self.initial_sigma = self.config.get("initial_sigma", 10.0)
        self.num_workers = self.config.get("num_workers", 8)
        self.use_hold = self.config.get("use_hold", True)
        self.lookahead = self.config.get("lookahead", 1)
        self.top_k = self.config.get("lookahead_top_k", 5)
        self.verbose = verbose

        # Initialize Gaussian distribution
        self.mu = np.zeros(self.num_features)
        self.sigma = np.ones(self.num_features) * self.initial_sigma

        self.best_weights = None
        self.best_score = 0

        logger.info(f"CEM config: pop={self.population_size}, elite={self.elite_count}, "
                     f"games={self.num_games}, gens={self.num_generations}, "
                     f"noise={self.noise_initial}→0 (decay={self.noise_decay}/gen), "
                     f"workers={self.num_workers}, "
                     f"hold={self.use_hold}, lookahead={self.lookahead}")

    def train(self, path: str):
        logger.info(f"Starting CEM optimization. Results saved to: {path}")
        os.makedirs(path, exist_ok=True)

        for gen in range(self.num_generations):
            # Sample weight vectors from Gaussian
            population = [np.random.normal(self.mu, self.sigma)
                          for _ in range(self.population_size)]

            # Evaluate in parallel
            eval_args = [(w, self.num_games, self.use_hold, self.lookahead, self.top_k)
                         for w in population]

            with Pool(self.num_workers) as pool:
                scores = list(tqdm(
                    pool.imap(evaluate_weights, eval_args),
                    total=self.population_size,
                    desc=f"Gen {gen+1}/{self.num_generations}",
                    disable=not self.verbose,
                ))

            scores = np.array(scores)

            # Select elite
            elite_idx = np.argsort(scores)[-self.elite_count:]
            elite_weights = np.array([population[i] for i in elite_idx])
            elite_scores = scores[elite_idx]

            # Update distribution (noisy CEM with decaying noise per Szita & Lőrincz)
            # Z_t = max(noise_initial - t * noise_decay, 0), added to variance
            noise_t = max(self.noise_initial - gen * self.noise_decay, 0.0)
            self.mu = np.mean(elite_weights, axis=0)
            elite_var = np.var(elite_weights, axis=0)
            self.sigma = np.sqrt(elite_var + noise_t)

            # Track best individual from this generation
            gen_best_idx = np.argmax(scores)
            gen_best = scores[gen_best_idx]
            if gen_best > self.best_score:
                self.best_score = gen_best
                self.best_weights = population[gen_best_idx].copy()
                np.save(os.path.join(path, "best_weights.npy"), self.best_weights)

            # Evaluate μ over 30 games every 5 generations (paper Section 3)
            # Parallelized across workers instead of sequential in 1 process
            mu_score = None
            if gen % 5 == 0 or gen == self.num_generations - 1:
                eval_mu_args = [(self.mu, 1, self.use_hold, self.lookahead, self.top_k)
                                for _ in range(30)]
                with Pool(self.num_workers) as pool:
                    mu_scores = pool.map(evaluate_weights, eval_mu_args)
                mu_score = np.mean(mu_scores)

            # Save checkpoint
            np.save(os.path.join(path, "mu.npy"), self.mu)
            np.save(os.path.join(path, "sigma.npy"), self.sigma)

            mu_str = f", μ_eval(30games)={mu_score:.0f}" if mu_score is not None else ""
            logger.info(
                f"Gen {gen+1}: best={gen_best:.0f}, elite_avg={elite_scores.mean():.0f}, "
                f"pop_avg={scores.mean():.0f}{mu_str}, "
                f"noise_Z={noise_t:.2f}"
            )
            logger.info(f"  μ = {np.round(self.mu, 3).tolist()}")
            logger.info(f"  σ = {np.round(self.sigma, 3).tolist()}")

        # Save final
        np.save(os.path.join(path, "final_weights.npy"), self.mu)
        logger.info(f"CEM complete. Best score: {self.best_score:.0f} lines")
        logger.info(f"Best weights: {np.round(self.best_weights, 4).tolist()}")

        return self.best_score

    def test(self, path: str, num_episodes: int = 10):
        """Test the best weights."""
        weights_path = os.path.join(path, "best_weights.npy")
        if os.path.exists(weights_path):
            weights = np.load(weights_path)
        elif self.best_weights is not None:
            weights = self.best_weights
        else:
            weights = self.mu

        logger.info(f"Testing weights: {np.round(weights, 4).tolist()}")

        results = []
        for ep in range(num_episodes):
            lines = play_game(weights, self.use_hold, self.lookahead, self.top_k)
            results.append(lines)
            if self.verbose:
                logger.info(f"  Game {ep+1}: {lines} lines")

        avg = np.mean(results)
        std = np.std(results)
        logger.info(f"Test results: {avg:.0f} ± {std:.0f} lines "
                     f"(min={min(results)}, max={max(results)})")
        return avg

    def close(self):
        pass
