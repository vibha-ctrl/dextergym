"""
Shadow Hand Dexterous Manipulation Environments
================================================

Tasks:
    - ButtonPress-v0: Press a red button

Usage:
    import gymnasium as gym
    import envs
    
    env = gym.make("ButtonPress-v0", render_mode="human")
"""

from gymnasium.envs.registration import register

register(
    id="ButtonPress-v0",
    entry_point="envs.button_press_env:ButtonPressEnv",
    max_episode_steps=200,
)

TASKS = [
    "ButtonPress-v0",
]
