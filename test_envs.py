#!/usr/bin/env python3
"""
Quick test script to verify environments work.

Usage:
    python test_envs.py
    python test_envs.py --render
"""

import argparse
import gymnasium as gym
import numpy as np

# Register environments
import envs
from envs import TASKS


def test_environment(task: str, render: bool = False, steps: int = 100):
    """Test a single environment."""
    print(f"\n🧪 Testing: {task}")
    
    try:
        # Create environment
        render_mode = "human" if render else None
        env = gym.make(task, render_mode=render_mode)
        
        # Check spaces
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        # Reset
        obs, info = env.reset()
        print(f"   Initial obs shape: {obs.shape}")
        
        # Run random actions
        total_reward = 0
        for i in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                obs, info = env.reset()
        
        env.close()
        print(f"   ✅ Passed! (Total reward over {steps} steps: {total_reward:.2f})")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test dexterous manipulation environments")
    parser.add_argument("--render", action="store_true", help="Render environments")
    parser.add_argument("--task", type=str, default=None, help="Test specific task only")
    args = parser.parse_args()
    
    print("="*50)
    print("🎮 Dexterous Benchmark - Environment Tests")
    print("="*50)
    
    tasks = [args.task] if args.task else TASKS
    results = {}
    
    for task in tasks:
        results[task] = test_environment(task, render=args.render)
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Summary")
    print("="*50)
    passed = sum(results.values())
    total = len(results)
    
    for task, passed_test in results.items():
        status = "✅" if passed_test else "❌"
        print(f"   {status} {task}")
    
    print(f"\n   Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All environments working!")
    else:
        print("\n⚠️ Some environments failed. Check errors above.")


if __name__ == "__main__":
    main()
