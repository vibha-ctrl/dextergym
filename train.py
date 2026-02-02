#!/usr/bin/env python3
"""
Training Script for Dexterous Manipulation Benchmark
====================================================

Train PPO policies on dexterous manipulation tasks.

Usage:
    # Train single task
    python train.py --task USBInsertion-v0
    
    # Train all tasks
    python train.py --task all
    
    # Custom timesteps
    python train.py --task CoinStack-v0 --timesteps 1000000
    
    # Resume training
    python train.py --task KeyTurn-v0 --checkpoint models/KeyTurn-v0/best_model.zip
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Register custom environments
import envs
from envs import TASKS


def make_env(env_id: str, rank: int, seed: int = 0):
    """Create a wrapped environment."""
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train_task(
    task: str,
    total_timesteps: int = 500_000,
    n_envs: int = 8,
    checkpoint: str = None,
    seed: int = 42,
    device: str = "auto",
):
    """
    Train a PPO policy on a task.
    
    Args:
        task: Environment ID (e.g., "USBInsertion-v0")
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        checkpoint: Path to checkpoint to resume from
        seed: Random seed
        device: Device to use ("cuda", "cpu", or "auto")
    """
    print(f"\n{'='*60}")
    print(f"🎯 Training: {task}")
    print(f"{'='*60}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Device: {device}")
    print(f"  Seed: {seed}")
    print()
    
    # Create directories
    log_dir = Path(f"logs/{task}")
    model_dir = Path(f"models/{task}")
    tb_log_dir = Path(f"tb_logs/{task}")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create parallel training environments
    env = SubprocVecEnv([make_env(task, i, seed) for i in range(n_envs)])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(task, 0, seed + 100)])
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(10000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix=task,
        verbose=0,
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Create or load model
    if checkpoint:
        print(f"📂 Loading checkpoint: {checkpoint}")
        model = PPO.load(checkpoint, env=env, device=device)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(tb_log_dir),
            device=device,
            seed=seed,
            verbose=1,
        )
    
    # Train
    print(f"\n🚀 Starting training...")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Save final model
    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))
    
    # Report
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Finished training {task}")
    print(f"   Time elapsed: {elapsed}")
    print(f"   Model saved to: {final_model_path}")
    print(f"   Best model at: {model_dir / 'best_model.zip'}")
    print(f"   TensorBoard logs: {tb_log_dir}")
    print(f"{'='*60}\n")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO on dexterous manipulation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py --task USBInsertion-v0
    python train.py --task all --timesteps 1000000
    python train.py --task CoinStack-v0 --n_envs 4 --device cpu
        """,
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="USBInsertion-v0",
        help=f"Task to train on. Options: {', '.join(TASKS + ['all'])}",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Determine tasks to train
    if args.task.lower() == "all":
        tasks_to_train = TASKS
    elif args.task in TASKS:
        tasks_to_train = [args.task]
    else:
        print(f"❌ Unknown task: {args.task}")
        print(f"   Available tasks: {', '.join(TASKS + ['all'])}")
        sys.exit(1)
    
    # Train
    print(f"\n🎮 Dexterous Manipulation Benchmark - Training")
    print(f"   Tasks: {', '.join(tasks_to_train)}")
    print(f"   Total: {len(tasks_to_train)} task(s)\n")
    
    for i, task in enumerate(tasks_to_train, 1):
        print(f"\n[{i}/{len(tasks_to_train)}] Training {task}...")
        train_task(
            task=task,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            checkpoint=args.checkpoint if len(tasks_to_train) == 1 else None,
            seed=args.seed,
            device=args.device,
        )
    
    print("\n🎉 All training complete!")
    print("   View training curves: tensorboard --logdir tb_logs/")


if __name__ == "__main__":
    main()
