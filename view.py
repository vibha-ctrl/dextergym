#!/usr/bin/env python3
"""
Viewer Script - Watch environments with random actions.

Usage:
    python view.py                          # View ButtonPress (default)
    python view.py --task CubeGrasp-v0      # View specific task
    python view.py --task all               # View all tasks
    python view.py --steps 1000             # Run for more steps
    python view.py --static                 # View static scene (no movement)
"""

import argparse
import time
import numpy as np
import gymnasium as gym

import envs
from envs import TASKS


def view_environment(task: str, steps: int = 500, static: bool = False):
    """View a single environment with random actions."""
    print(f"\nViewing: {task}")
    if static:
        print("Mode: STATIC (no movement)")
    else:
        print(f"Steps: {steps}")
    print("Controls: Close window or Ctrl+C to stop\n")
    
    env = gym.make(task, render_mode="human")
    obs, info = env.reset()
    
    # Stabilize hand position right after reset (run a few steps)
    # Base position is now in XML, so action=0 keeps hand in place
    stable_action = np.zeros(env.action_space.shape)
    for _ in range(5):
        env.step(stable_action)
    
    try:
        if static:
            # Static mode: just render without taking actions
            print("Press Ctrl+C to exit...")
            while True:
                env.render()
                time.sleep(0.05)
        else:
            for step in range(steps):
                # Keep base position stable (first 6 actuators), only randomize fingers
                action = env.action_space.sample()
                # Let base position move randomly, lock rotations
                action[3:6] = 0.0  # Lock rotations
                if step % 50 == 0:  # Print less frequently
                    data = env.unwrapped.data
                    model = env.unwrapped.model
                    joint_id = model.joint('button_joint').id
                    qpos_addr = model.jnt_qposadr[joint_id]
                    button_qpos = data.qpos[qpos_addr]
                    status = "PRESSED!" if button_qpos < -0.001 else ""
                    print(f"Button: {button_qpos:.4f} {status}")
                obs, reward, terminated, truncated, info = env.step(action)
                time.sleep(0.02)
                
                if terminated or truncated:
                    print(f"Episode ended at step {step}. Resetting...")
                    obs, info = env.reset()
                    # Stabilize - base position is in XML, action=0 keeps it there
                    stable_action = np.zeros(env.action_space.shape)
                    for _ in range(5):
                        env.step(stable_action)
                
    except KeyboardInterrupt:
        print("\nStopped by user.")
    
    env.close()
    print(f"Done viewing {task}\n")


def main():
    parser = argparse.ArgumentParser(
        description="View dexterous manipulation environments"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ButtonPress-v0",
        help=f"Task to view. Options: {', '.join(TASKS + ['all'])}",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of steps to run (default: 500)",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="View static scene without any movement",
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Shadow Hand Dexterous Manipulation Viewer")
    print("=" * 50)
    
    if args.task.lower() == "all":
        for task in TASKS:
            view_environment(task, args.steps, args.static)
            print("Press Enter for next task (or Ctrl+C to quit)...")
            try:
                input()
            except KeyboardInterrupt:
                break
    elif args.task in TASKS:
        view_environment(args.task, args.steps, args.static)
    else:
        print(f"Unknown task: {args.task}")
        print(f"Available: {', '.join(TASKS)}")


if __name__ == "__main__":
    main()
