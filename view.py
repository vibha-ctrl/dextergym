#!/usr/bin/env python3
"""
Viewer Script - Watch environments with random actions.

Usage:
    python view.py                          # View USBInsertion (default)
    python view.py --task CoinStack-v0      # View specific task
    python view.py --task all               # View all tasks one by one
    python view.py --steps 1000             # Run for more steps
"""

import argparse
import time
import gymnasium as gym

# Register environments
import envs
from envs import TASKS


def view_environment(task: str, steps: int = 500):
    """View a single environment with random actions."""
    print(f"\n🎮 Viewing: {task}")
    print(f"   Steps: {steps}")
    print(f"   Controls: Close window or Ctrl+C to stop\n")
    
    # Create environment with rendering
    env = gym.make(task, render_mode="human")
    
    obs, info = env.reset()
    
    try:
        for step in range(steps):
            # Small random action (scaled down for smoother movement)
            action = env.action_space.sample() * 0.3
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Slower playback
            time.sleep(0.05)
            
            if terminated or truncated:
                print(f"   Episode ended at step {step}. Resetting...")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\n   Stopped by user.")
    
    env.close()
    print(f"✅ Done viewing {task}\n")


def main():
    parser = argparse.ArgumentParser(
        description="View dexterous manipulation environments"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="USBInsertion-v0",
        help=f"Task to view. Options: {', '.join(TASKS + ['all'])}",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of steps to run (default: 500)",
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("🤖 Dexterous Benchmark - Environment Viewer")
    print("="*50)
    
    if args.task.lower() == "all":
        for task in TASKS:
            view_environment(task, args.steps)
            print("Press Enter for next task (or Ctrl+C to quit)...")
            try:
                input()
            except KeyboardInterrupt:
                break
    elif args.task in TASKS:
        view_environment(args.task, args.steps)
    else:
        print(f"❌ Unknown task: {args.task}")
        print(f"   Available: {', '.join(TASKS)}")


if __name__ == "__main__":
    main()
