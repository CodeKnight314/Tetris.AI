import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

BOARD_HEIGHT = 20
BOARD_WIDTH = 10

PIECE_SHAPES = {
    0: [  # I
        np.array([[1, 1, 1, 1]], dtype=np.uint8),
        np.array([[1], [1], [1], [1]], dtype=np.uint8),
    ],
    1: [  # O
        np.array([[1, 1], [1, 1]], dtype=np.uint8),
    ],
    2: [  # T
        np.array([[0, 1, 0], [1, 1, 1]], dtype=np.uint8),
        np.array([[0, 1], [1, 1], [0, 1]], dtype=np.uint8),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=np.uint8),
    ],
    3: [  # S
        np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8),
        np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8),
    ],
    4: [  # Z
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8),
        np.array([[0, 1], [1, 1], [1, 0]], dtype=np.uint8),
    ],
    5: [  # J
        np.array([[1, 0, 0], [1, 1, 1]], dtype=np.uint8),
        np.array([[0, 1], [0, 1], [1, 1]], dtype=np.uint8),
        np.array([[1, 1, 1], [0, 0, 1]], dtype=np.uint8),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=np.uint8),
    ],
    6: [  # L
        np.array([[0, 0, 1], [1, 1, 1]], dtype=np.uint8),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=np.uint8),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=np.uint8),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=np.uint8),
    ],
}

NUM_FEATURES = 22  # Bertsekas & Tsitsiklis (1996): 10 heights + 9 diffs + max_height + holes + bias


@dataclass
class Placement:
    piece_id: int
    rotation_idx: int
    column: int
    drop_row: int
    shape: np.ndarray
    use_hold: bool
    afterstate_board: np.ndarray
    lines_cleared: int
    features: np.ndarray
    landing_height: float
    piece_cells_in_cleared: int = 0


def find_drop_row(board: np.ndarray, shape: np.ndarray, col: int) -> Optional[int]:
    """Find the lowest row where `shape` can be placed at `col` without overlap.

    Returns None if the piece cannot be placed (collision at row 0).
    """
    h, w = shape.shape
    if col + w > BOARD_WIDTH or col < 0:
        return None

    piece_mask = shape > 0
    board_region = board[:, col:col + w] > 0  # (20, w) bool

    for row in range(BOARD_HEIGHT - h + 1):
        if np.any(piece_mask & board_region[row:row + h]):
            return row - 1 if row > 0 else None
        # Piece fits at this row, continue to find lowest
    return BOARD_HEIGHT - h


def simulate_afterstate(board: np.ndarray, shape: np.ndarray, col: int, drop_row: int):
    """Place piece and clear lines. Returns (afterstate_board, lines_cleared, piece_cells_in_cleared)."""
    afterstate = board.copy()
    h, w = shape.shape

    # Place piece cells and track their positions
    piece_positions = []
    for r in range(h):
        for c in range(w):
            if shape[r, c]:
                afterstate[drop_row + r, col + c] = 255
                piece_positions.append(drop_row + r)

    # Find full rows
    full_rows = np.all(afterstate == 255, axis=1)
    lines_cleared = int(np.sum(full_rows))

    # Count individual piece cells in cleared rows
    piece_cells_in_cleared = sum(1 for row in piece_positions if full_rows[row])

    # Clear lines
    if lines_cleared > 0:
        remaining = afterstate[~full_rows]
        empty_rows = np.zeros((lines_cleared, BOARD_WIDTH), dtype=np.uint8)
        afterstate = np.vstack([empty_rows, remaining])

    return afterstate, lines_cleared, piece_cells_in_cleared


def compute_bertsekas_features(board: np.ndarray) -> np.ndarray:
    """Compute 22 Bertsekas & Tsitsiklis (1996) features on a board state.

    Unnormalized raw values, matching the paper exactly:
      [h1..h10, |h1-h2|..|h9-h10|, max_height, holes, bias]

    Returns float64 array of shape (22,).
    """
    binary = board > 0
    height, width = BOARD_HEIGHT, BOARD_WIDTH

    # Column heights (0 = empty column, max = 20)
    col_has_filled = np.any(binary, axis=0)
    first_filled = np.argmax(binary, axis=0)
    col_heights = np.where(col_has_filled, height - first_filled, 0).astype(np.float64)

    # Absolute height differences between adjacent columns
    height_diffs = np.abs(np.diff(col_heights))

    # Max column height
    max_height = np.max(col_heights)

    # Holes: empty cells with at least one filled cell above in same column
    cummax = np.maximum.accumulate(binary, axis=0)
    holes = float(np.sum(cummax & ~binary))

    # Bias term
    bias = 1.0

    # 10 heights + 9 diffs + max_height + holes + bias = 22
    return np.concatenate([col_heights, height_diffs, [max_height, holes, bias]])


def enumerate_placements(board: np.ndarray, piece_id: int, use_hold: bool = False) -> List[Placement]:
    """Enumerate all valid placements for a piece on the board.

    Args:
        board: 20x10 binary board (0=empty, 255=filled)
        piece_id: piece type index 0-6
        use_hold: whether this placement requires a hold swap

    Returns:
        List of Placement objects with precomputed afterstates and features.
    """
    placements = []
    for rot_idx, shape in enumerate(PIECE_SHAPES[piece_id]):
        h, w = shape.shape
        for col in range(BOARD_WIDTH - w + 1):
            drop_row = find_drop_row(board, shape, col)
            if drop_row is None:
                continue

            afterstate, lines_cleared, piece_cells = simulate_afterstate(
                board, shape, col, drop_row
            )
            landing_height = BOARD_HEIGHT - drop_row - h / 2.0

            features = compute_bertsekas_features(afterstate)

            placements.append(Placement(
                piece_id=piece_id,
                rotation_idx=rot_idx,
                column=col,
                drop_row=drop_row,
                shape=shape,
                use_hold=use_hold,
                afterstate_board=afterstate,
                lines_cleared=lines_cleared,
                features=features,
                landing_height=landing_height,
                piece_cells_in_cleared=piece_cells,
            ))

    return placements


def enumerate_placements_with_hold(board: np.ndarray, current_piece_id: int,
                                   hold_piece_id: Optional[int],
                                   next_piece_id: int,
                                   has_swapped: bool) -> List[Placement]:
    """Enumerate placements for current piece and optionally the hold piece."""
    placements = enumerate_placements(board, current_piece_id, use_hold=False)

    if not has_swapped:
        swap_piece_id = hold_piece_id if hold_piece_id is not None else next_piece_id
        if swap_piece_id != current_piece_id:  # skip if same piece type
            hold_placements = enumerate_placements(board, swap_piece_id, use_hold=True)
            placements.extend(hold_placements)

    return placements
