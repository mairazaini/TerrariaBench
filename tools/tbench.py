#!/usr/bin/env python3
"""
tools/tbench.py - CLI Bridge for TerrariaBench & Harbor / OpenCode Interaction

Allows agents or bash harnesses to query the environment, step actions, and lookup mechanics:
  python tools/tbench.py observe
  python tools/tbench.py action --type click --x 100 --y 150
  python tools/tbench.py wiki --query "housing requirements"
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp_server.terraria_mcp import observe_environment, execute_terraria_action, lookup_terraria_mechanics


def main():
    parser = argparse.ArgumentParser(description="TerrariaBench CLI Bridge")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # observe
    obs_p = subparsers.add_parser("observe", help="Capture observation and state header")
    obs_p.add_argument("--no-grid", action="store_true", help="Disable coordinate grid overlay")

    # action
    act_p = subparsers.add_parser("action", help="Execute an action")
    act_p.add_argument("--type", type=str, required=True, help="Action type: click, move, key, mouse_down, batch")
    act_p.add_argument("--x", type=int, default=0, help="X pixel coordinate")
    act_p.add_argument("--y", type=int, default=0, help="Y pixel coordinate")
    act_p.add_argument("--key", type=str, default="", help="Keyboard key to press")
    act_p.add_argument("--duration", type=int, default=100, help="Duration in milliseconds")

    # wiki
    wiki_p = subparsers.add_parser("wiki", help="Lookup Terraria mechanics (Context.dev & Oxylabs)")
    wiki_p.add_argument("--query", type=str, required=True, help="Mechanics query string")

    args = parser.parse_args()

    if args.command == "observe":
        res = observe_environment(include_grid=not args.no_grid)
        print(f"State Header: {res.get('state_header')}")
        print(f"Status: {res.get('status')} | Frame: {res.get('frame_path')}")
        print(json.dumps(res.get("game_state", {}), indent=2))

    elif args.command == "action":
        res = execute_terraria_action(
            action_type=args.type,
            x=args.x,
            y=args.y,
            key=args.key,
            duration_ms=args.duration
        )
        print(json.dumps(res, indent=2))

    elif args.command == "wiki":
        content = lookup_terraria_mechanics(args.query)
        print(content)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
