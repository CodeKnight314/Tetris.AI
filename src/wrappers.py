import gymnasium as gym
import numpy as np
from src.placement import BOARD_HEIGHT, BOARD_WIDTH


class PlacementEnv(gym.Wrapper):
    """Wraps tetris_gymnasium/Tetris for placement-based RL.

    The underlying env should be created with gravity disabled.
    """
    def __init__(self, env):
        super().__init__(env)
        self.tetris = env.unwrapped
        assert not self.tetris.gravity_enabled, \
            "PlacementEnv requires gravity=False. Create env with Tetris(gravity=False)."
        self.padding = self.tetris.padding

    def get_board_binary(self) -> np.ndarray:
        """Extract 20x10 binary board (without active piece).

        Returns uint8 array: 0=empty, 255=filled.
        """
        raw = self.tetris.board
        cropped = raw[0:self.tetris.height, self.padding:self.padding + self.tetris.width]
        return np.where(cropped == 0, 0, 255).astype(np.uint8)

    def _to_piece_index(self, piece) -> int:
        """Convert a tetromino object or raw ID to a 0-6 piece index."""
        if hasattr(piece, 'id'):
            # Tetromino object — ID is offset by base_pixels (empty + bedrock)
            return int(piece.id) - len(self.tetris.base_pixels)
        else:
            # Raw integer from queue — already 0-indexed
            return int(piece)

    def get_piece_id(self) -> int:
        """Get current active piece index (0-6)."""
        return self._to_piece_index(self.tetris.active_tetromino)

    def get_hold_piece_id(self):
        """Get held piece index (0-6) or None if hold is empty."""
        held = self.tetris.holder.get_tetrominoes()
        if len(held) == 0:
            return None
        return self._to_piece_index(held[0])

    def get_next_piece_id(self) -> int:
        """Peek at the next piece in queue (0-6)."""
        queue = self.tetris.queue.get_queue()
        return self._to_piece_index(queue[0])

    def get_has_swapped(self) -> bool:
        return self.tetris.has_swapped

    def execute_placement(self, placement):
        """Execute a placement using low-level actions.

        Sequence: [swap] → rotate × N → move to column → hard_drop.
        Gravity must be disabled for safe positioning.

        Returns (obs, total_reward, terminated, truncated, info) from the final hard_drop.
        """
        total_reward = 0.0

        # Hold swap if needed
        if placement.use_hold:
            obs, r, term, trunc, info = self.env.step(6)  # swap
            total_reward += r
            if term or trunc:
                return obs, total_reward, term, trunc, info

        # Rotate to target rotation
        for _ in range(placement.rotation_idx):
            obs, r, term, trunc, info = self.env.step(3)  # rotate_cw
            total_reward += r
            if term or trunc:
                return obs, total_reward, term, trunc, info

        # Compute column delta
        current_col = self._get_current_left_col()
        target_col = placement.column
        delta = target_col - current_col

        # Move to target column
        if delta > 0:
            for _ in range(delta):
                obs, r, term, trunc, info = self.env.step(1)  # move_right
                total_reward += r
                if term or trunc:
                    return obs, total_reward, term, trunc, info
        elif delta < 0:
            for _ in range(-delta):
                obs, r, term, trunc, info = self.env.step(0)  # move_left
                total_reward += r
                if term or trunc:
                    return obs, total_reward, term, trunc, info

        # Hard drop
        obs, r, term, trunc, info = self.env.step(5)  # hard_drop
        total_reward += r

        return obs, total_reward, term, trunc, info

    def _get_current_left_col(self) -> int:
        """Get the leftmost filled column of the active piece in 0-9 board coords."""
        matrix = self.tetris.active_tetromino.matrix
        # Find leftmost filled column in the piece matrix
        left_offset = 0
        for c in range(matrix.shape[1]):
            if np.any(matrix[:, c] > 0):
                left_offset = c
                break
        # Convert to board coords (unpadded)
        return self.tetris.x + left_offset - self.padding

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
