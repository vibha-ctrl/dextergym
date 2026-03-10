"""
Shadow Hand Dexterous Manipulation Environments
================================================

Tasks:
    - ButtonPress-v0: Press a red button
    - CubePush-v0: Push a cube into a target ring
    - CubePickup-v0: Grasp and lift a cube to a target height

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

register(
    id="CubePush-v0",
    entry_point="envs.cube_push_env:CubePushEnv",
    max_episode_steps=500,
)

register(
    id="CubePickup-v0",
    entry_point="envs.cube_pickup_env:CubePickupEnv",
    max_episode_steps=500,
)

TASKS = [
    "ButtonPress-v0",
    "CubePush-v0",
    "CubePickup-v0",
]
