from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from bomb_game import LLMAgentController, DeepSeekLLMClient
from bomb_game.game import GameManager, create_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query DeepSeek for multi-agent ToM plans")
    parser.add_argument("--game-id", type=str, default=None, help="Existing game identifier to resume")
    parser.add_argument("--api-key", type=str, default=None, help="DeepSeek API key (falls back to env var)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager: GameManager = create_manager()
    if args.game_id:
        state = manager.get_game(args.game_id)
    else:
        state = manager.create_game()
    client = DeepSeekLLMClient(api_key=args.api_key)
    controller = LLMAgentController(client)
    plans = controller.propose_joint_actions(state)
    print(json.dumps(plans, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
