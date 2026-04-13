import numpy as np
from src.placement import enumerate_placements, NUM_FEATURES

class LinearAgent:
    def __init__(self, weights: np.ndarray):
        self.weights = np.array(weights, dtype=np.float64)

    def evaluate(self, features: np.ndarray) -> float:
        return np.dot(self.weights, features)

    def select_placement(self, placements, next_piece_id=None,
                         lookahead=1, top_k=5):
        """Select the best placement, optionally with 2-piece lookahead.

        Args:
            placements: list of Placement objects
            next_piece_id: piece ID for lookahead (0-6), needed if lookahead >= 2
            lookahead: 1 = greedy on current piece, 2 = also search next piece
            top_k: number of top candidates to evaluate with lookahead

        Returns:
            Index of the best placement, or None if no placements.
        """
        if not placements:
            return None

        scores = np.array([self.evaluate(p.features) for p in placements])

        if lookahead <= 1 or next_piece_id is None:
            return int(np.argmax(scores))

        # 2-piece lookahead: re-evaluate top-K with next piece search
        k = min(top_k, len(placements))
        top_indices = np.argpartition(scores, -k)[-k:]

        best_score = -np.inf
        best_idx = top_indices[0]

        for ci in top_indices:
            cp = placements[ci]
            next_placements = enumerate_placements(cp.afterstate_board, next_piece_id)
            if next_placements:
                best_next = max(self.evaluate(np_.features) for np_ in next_placements)
                score = scores[ci] + best_next
            else:
                score = scores[ci]

            if score > best_score:
                best_score = score
                best_idx = ci

        return int(best_idx)

    def save_weights(self, path: str):
        np.save(path, self.weights)

    @classmethod
    def load_weights(cls, path: str):
        weights = np.load(path)
        return cls(weights)
