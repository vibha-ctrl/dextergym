#!/usr/bin/env python3
"""
Training Script for Dexterous Manipulation Benchmark
====================================================

Train PPO policies on dexterous manipulation tasks.

Usage:
    # Train single task
    python train.py --task ButtonPress-v0
    
    # Train all tasks
    python train.py --task all
    
    # Custom timesteps
    python train.py --task ButtonPress-v0 --timesteps 1000000
    
    # Resume training
    python train.py --task ButtonPress-v0 --checkpoint models/ButtonPress-v0/best_model.zip
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
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


class SuccessRateEvalCallback(EvalCallback):
    """EvalCallback that saves best model by success rate instead of mean reward."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_success_rate = -1.0
    
    def _on_step(self) -> bool:
        # Run parent's _on_step which does evaluation, logging, etc.
        # But we override the "best model" saving logic
        continue_training = True
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset success buffer
            self._is_success_buffer = []
            
            from stable_baselines3.common.evaluation import evaluate_policy
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )
            
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)
            
            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            
            # Log to tensorboard
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            
            # Compute and log success rate
            success_rate = 0.0
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)
            
            self.logger.dump(self.num_timesteps)
            
            # Save best model by SUCCESS RATE (not mean reward)
            if success_rate > self.best_success_rate:
                if self.verbose >= 1:
                    print(f"New best success rate: {100*success_rate:.1f}%")
                self.best_success_rate = success_rate
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            elif success_rate == self.best_success_rate and mean_reward > getattr(self, 'best_mean_reward', -np.inf):
                # Tie-break: same success rate, higher reward wins
                if self.verbose >= 1:
                    print(f"New best mean reward (same success rate {100*success_rate:.1f}%)!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            
            self.best_mean_reward = max(getattr(self, 'best_mean_reward', -np.inf), float(mean_reward))
            
            # Store eval results
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)
            
            kwargs = {}
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)
            
            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                **kwargs,
            )
            
            # Trigger callbacks
            if self.callback_on_new_best is not None and success_rate >= self.best_success_rate:
                continue_training = self.callback_on_new_best.on_step()
            if self.callback is not None:
                continue_training = continue_training and self._on_event()
        
        return continue_training

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
    
    # Callbacks — saves best model by SUCCESS RATE, not mean reward
    eval_callback = SuccessRateEvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=max(10000 // n_envs, 1),
        n_eval_episodes=20,
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
        # Linear learning rate decay: starts at 2e-4, decays to 1e-5
        lr_schedule = lambda progress: 1e-5 + (2e-4 - 1e-5) * progress

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=lr_schedule,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.15,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.005,
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
    python train.py --task ButtonPress-v0
    python train.py --task all --timesteps 1000000
    python train.py --task ButtonPress-v0 --n_envs 4 --device cpu
        """,
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default="ButtonPress-v0",
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
