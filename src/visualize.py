import argparse
import numpy as np
from tetris_gymnasium.envs import Tetris
from gymnasium.wrappers import RecordVideo
from src.agent import LinearAgent
from src.wrappers import PlacementEnv
from src.placement import enumerate_placements
from tqdm import tqdm

def play_and_record(weights_path: str, output_dir: str, max_pieces: int = 500,
                    fps: int = 6, upscale: int = 20):
    """Play a game with the given weights and record to video.

    Args:
        weights_path: Path to .npy weights file.
        output_dir: Directory to save the video.
        max_pieces: Maximum pieces to place before stopping.
        fps: Playback frames per second.
        upscale: Pixel upscale factor for the render.
    """
    weights = np.load(weights_path)
    agent = LinearAgent(weights)

    base_env = Tetris(render_mode="rgb_array", gravity=False, render_upscale=upscale)
    rec_env = RecordVideo(base_env, video_folder=output_dir,
                          episode_trigger=lambda _: True,
                          name_prefix="tetris", fps=fps)
    env = PlacementEnv(rec_env)
    env.reset()
    board = env.get_board_binary()

    total_lines = 0
    pieces = 0
    
    pbar = tqdm(total=max_pieces, desc="Playing game", disable=max_pieces == 0)
    while max_pieces == 0 or pieces < max_pieces:
        if max_pieces != 0:
            pbar.update(1)
        piece_id = env.get_piece_id()
        placements = enumerate_placements(board, piece_id)

        if not placements:
            break

        idx = agent.select_placement(placements)
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
    print(f"Done: {pieces} pieces, {total_lines} lines cleared")
    print(f"Video saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained Tetris agent")
    parser.add_argument("--weights", type=str, default="models/best_weights.npy",
                        help="Path to weights .npy file")
    parser.add_argument("--output", type=str, default="./videos",
                        help="Output directory for video")
    parser.add_argument("--max_pieces", type=int, default=0,
                        help="Max pieces to place (0 = no limit)")
    parser.add_argument("--fps", type=int, default=6,
                        help="Video playback FPS")
    parser.add_argument("--upscale", type=int, default=20,
                        help="Pixel upscale factor")
    args = parser.parse_args()

    play_and_record(args.weights, args.output, args.max_pieces, args.fps,
                    args.upscale)
