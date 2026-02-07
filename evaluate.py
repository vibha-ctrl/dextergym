#!/usr/bin/env python3
"""
Evaluation Script for Dexterous Manipulation Benchmark
======================================================

Evaluate trained policies and visualize their performance.

Usage:
    # Evaluate with rendering
    python evaluate.py --task ButtonPress-v0 --render
    
    # Evaluate all tasks
    python evaluate.py --task all --episodes 50
    
    # Evaluate specific checkpoint
    python evaluate.py --task ButtonPress-v0 --model models/ButtonPress-v0/best_model.zip
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import time

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from tqdm import tqdm

# Register custom environments
import envs
from envs import TASKS


def evaluate_policy(
    task: str,
    model_path: str = None,
    n_episodes: int = 20,
    render: bool = False,
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a trained policy.
    
    Args:
        task: Environment ID
        model_path: Path to trained model (default: models/{task}/best_model.zip)
        n_episodes: Number of evaluation episodes
        render: Whether to render
        deterministic: Use deterministic actions
        verbose: Print progress
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Default model path
    if model_path is None:
        model_path = f"models/{task}/best_model.zip"
    
    # Check model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Train first with: python train.py --task {task}")
        return None
    
    # Create environment
    render_mode = "human" if render else None
    env = gym.make(task, render_mode=render_mode)
    
    # Load model
    model = PPO.load(model_path)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"📊 Evaluating: {task}")
        print(f"{'='*50}")
        print(f"   Model: {model_path}")
        print(f"   Episodes: {n_episodes}")
        print(f"   Render: {render}")
        print()
    
    # Evaluation metrics
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    successes: List[bool] = []
    
    # Run episodes
    iterator = range(n_episodes)
    if verbose and not render:
        iterator = tqdm(iterator, desc="Evaluating")
    
    for ep in iterator:
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if render:
                time.sleep(0.01)  # Slow down for visualization
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        successes.append(info.get("is_success", False))
        
        if verbose and render:
            status = "✅" if info.get("is_success", False) else "❌"
            print(f"  Episode {ep+1}: Reward={episode_reward:.1f}, Steps={episode_length}, {status}")
    
    env.close()
    
    # Compute statistics
    results = {
        "task": task,
        "model": model_path,
        "episodes": n_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": np.mean(successes) * 100,
        "successes": sum(successes),
    }
    
    if verbose:
        print(f"\n📈 Results:")
        print(f"   Mean Reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   Mean Length:  {results['mean_length']:.1f} steps")
        print(f"   Success Rate: {results['success_rate']:.1f}% ({results['successes']}/{n_episodes})")
        print()
    
    return results


def evaluate_all_tasks(
    n_episodes: int = 20,
    deterministic: bool = True,
) -> List[Dict]:
    """Evaluate all tasks and print summary."""
    all_results = []
    
    print(f"\n{'='*70}")
    print(f"🎮 Dexterous Manipulation Benchmark - Full Evaluation")
    print(f"{'='*70}\n")
    
    for task in TASKS:
        result = evaluate_policy(
            task=task,
            n_episodes=n_episodes,
            render=False,
            deterministic=deterministic,
            verbose=True,
        )
        if result:
            all_results.append(result)
    
    # Print summary table
    if all_results:
        print(f"\n{'='*70}")
        print(f"📊 BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"{'Task':<20} {'Reward':>12} {'Success':>12} {'Avg Steps':>12}")
        print("-" * 70)
        
        for r in all_results:
            print(f"{r['task']:<20} {r['mean_reward']:>10.1f} {r['success_rate']:>10.1f}% {r['mean_length']:>10.1f}")
        
        # Overall stats
        avg_success = np.mean([r['success_rate'] for r in all_results])
        print("-" * 70)
        print(f"{'AVERAGE':<20} {'':<12} {avg_success:>10.1f}%")
        print(f"{'='*70}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained policies on dexterous manipulation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="ButtonPress-v0",
        help=f"Task to evaluate. Options: {', '.join(TASKS + ['all'])}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (default: models/{task}/best_model.zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic (non-deterministic) actions",
    )
    
    args = parser.parse_args()
    
    if args.task.lower() == "all":
        if args.render:
            print("⚠️ Render mode disabled for 'all' tasks evaluation")
        evaluate_all_tasks(
            n_episodes=args.episodes,
            deterministic=not args.stochastic,
        )
    elif args.task in TASKS:
        evaluate_policy(
            task=args.task,
            model_path=args.model,
            n_episodes=args.episodes,
            render=args.render,
            deterministic=not args.stochastic,
        )
    else:
        print(f"❌ Unknown task: {args.task}")
        print(f"   Available tasks: {', '.join(TASKS + ['all'])}")
        sys.exit(1)


if __name__ == "__main__":
    main()
