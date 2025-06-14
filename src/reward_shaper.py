from typing import Tuple
import numpy as np
import logging
logger = logging.getLogger(__name__)

class RewardShaping():
    def __init__(self, weights: Tuple[int]): 
        self.weights = np.array(weights)
    
    def extract_features(self, board: np.array):
        height, width = board.shape
        holes = 0
        heights = np.zeros(width, dtype=int)
        
        for col in range(width):
            column = board[:, col]
            filled = np.where(column == 255)[0]
            if filled.size > 0:
                first_filled = filled[0]
                heights[col] = height - first_filled
                holes += np.sum(column[first_filled:] == 0)
            else:
                heights[col] = 0
                
        bumpiness = np.sum(np.abs(np.diff(heights)))
        aggregate_height = np.sum(heights)
        
        lines_cleared = np.sum(np.all(board == 255, axis=1))
        tetris_bonus = 1 if lines_cleared == 4 else 0

        return np.array([
            lines_cleared,
            holes,
            bumpiness, 
            aggregate_height,
            tetris_bonus
        ])
        
    def calculate_rewards(self, board: np.array):
        features = self.extract_features(board)
        if features[0] != 0:
            logger.info(f"Lines Cleared: {features[0]}")
        return np.dot(self.weights, features[:-1]) + features[-1]