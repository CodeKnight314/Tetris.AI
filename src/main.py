import argparse
from src.env import CEMTrainer

def main(args):
    trainer = CEMTrainer(args.c, args.verbose)
    try:
        if args.mode == "train":
            trainer.train(args.o)
        elif args.mode == "test":
            trainer.test(args.o, args.num_episodes)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        if trainer.best_weights is not None:
            import numpy as np, os
            os.makedirs(args.o, exist_ok=True)
            np.save(os.path.join(args.o, "interrupted_weights.npy"), trainer.best_weights)
            print(f"Best weights saved to {args.o}/interrupted_weights.npy")
        trainer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tetris AI — CEM Weight Optimization")
    parser.add_argument("--c", type=str, required=True, help="Config YAML path")
    parser.add_argument("--o", type=str, required=True, help="Output directory")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)
