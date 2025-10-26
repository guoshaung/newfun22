from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from bomb_game import BombDefusalEnv, MAPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MAPPO baseline on the bomb defusal game")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes to run")
    parser.add_argument("--rollout-length", type=int, default=32, help="Number of environment steps per rollout")
    parser.add_argument("--output", type=Path, default=Path("mappo_metrics.json"), help="Where to write aggregated metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = BombDefusalEnv()
    trainer = MAPPOTrainer(env)

    history: list[dict[str, float]] = []
    for episode in range(1, args.episodes + 1):
        last_obs = trainer.collect_rollout(args.rollout_length)
        metrics = trainer.update(last_obs)
        metrics["episode"] = episode
        history.append(metrics)
        print(f"Episode {episode}: {metrics}")

    args.output.write_text(json.dumps(history, indent=2))
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
