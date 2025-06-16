from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RewardShaping():
    def __init__(self, weights: Tuple[int]): 
        self.weights = np.array(weights)
    
    def extract_features(self, board: np.array, terminated: bool):
        height, width = board.shape

        lines_cleared = self.calculate_lines_cleared(board)
        holes = self.calculate_holes(board) / (height * width)
        aggregate_height, bumpiness, max_height = self.calculate_height(board)

        aggregate_height = aggregate_height / (height * width)
        bumpiness = bumpiness / (height * width / 2)
        max_height = max_height / height
        
        line_progress = self.calculate_line_progress(board)
        low_placement = self.calculate_low_placement_bonus(board)
        
        tetris_bonus = 1.0 if lines_cleared == 4 else 0
        survival_bonus = 1.0 if not terminated else 0.0
        
        reward_dict = {
            "lines_cleared": lines_cleared,
            "holes": -holes,
            "bumpiness": -bumpiness, 
            "aggregate_height": -aggregate_height,
            "max_height": -max_height, 
            "tetris_bonus": tetris_bonus, 
            "survival_bonus": survival_bonus,
            "line_progress": line_progress,
            "low_placement": low_placement
        }

        return np.clip(np.array([
            lines_cleared,
            -holes,
            -bumpiness, 
            -aggregate_height,
            -max_height, 
            tetris_bonus, 
            survival_bonus,
            line_progress,
            low_placement
        ]), -1.0, 1.0), reward_dict
    
    def calculate_line_progress(self, board: np.array):
        progress = 0
        for row in range(board.shape[0]):
            filled_cells = np.sum(board[row] == 255)
            if filled_cells > 0:
                progress += (filled_cells / board.shape[1]) ** 2
        return progress
    
    def calculate_low_placement_bonus(self, board: np.array):
        filled_positions = np.where(board == 255)
        if len(filled_positions[0]) == 0:
            return 0

        center_of_mass_row = np.mean(filled_positions[0])
        return (center_of_mass_row / board.shape[0])
        
    def calculate_holes(self, board: np.array):
        holes = 0
        for col in range(board.shape[1]):
            for row in range(board.shape[0]-1):
                if board[row, col] == 255 and board[row+1, col] == 0 and self.has_path_to_bottom(board, row, col):
                    holes += board.shape[0] - row
        return holes
    
    def calculate_lines_cleared(self, board: np.array):
        return np.sum(np.all(board == 255, axis=1))
    
    def calculate_height(self, board: np.array):
        heights = np.zeros(board.shape[1])
        for col in range(board.shape[1]):
            highest_row = None
            for row in range(board.shape[0]):
                if board[row, col] == 255:
                    highest_row = row
                    break
            
            if highest_row is None:
                heights[col] = 0
                continue
            
            if self.has_path_to_bottom(board, highest_row, col):
                heights[col] = board.shape[0] - highest_row
            else:
                heights[col] = 0
    
        aggregate_height = sum(heights)
        bumpiness = sum(np.array([abs(heights[i] - heights[i+1]) for i in range(len(heights)-1)]))
        max_height = np.max(heights)
        return aggregate_height, bumpiness, max_height
        
    def calculate_rewards(self, board: np.array, terminated: bool):
        features, reward_dict = self.extract_features(board, terminated)
        if features[0] != 0:
            logger.info(f"Lines Cleared: {features[0]}")
        reward_sum = np.dot(self.weights, features)
        reward_dict["sum"] = reward_sum
        return reward_sum, reward_dict
    
    def has_path_to_bottom(self, board: np.array, start_row: int, start_column: int):
        height, width = board.shape
        visited = set()
        stack = [(start_row, start_column)]
        directions = [(1, 0), (1, -1), (1, 1)]
        
        while stack: 
            row, col = stack.pop()
            
            if row == height: 
                return True
            
            if (row, col) in visited or row < 0 or row >= height or col < 0 or col >= width:
                continue
            
            if board[row, col] != 255: 
                continue 
            
            for dir in directions: 
                delta_y, delta_x = dir 
                new_row = delta_y + row
                new_col = delta_x + col
                if (new_row, new_col) not in visited: 
                    stack.append((new_row, new_col))
                    
        return False