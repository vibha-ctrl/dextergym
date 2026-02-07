"""
Shadow Hand Dexterous Manipulation Environments
================================================

Tasks:
    - ButtonPress-v0: Press a red button
    - CubeGrasp-v0: Grasp and lift a cube

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
    id="CubeGrasp-v0",
    entry_point="envs.cube_grasp_env:CubeGraspEnv",
    max_episode_steps=200,
)

TASKS = [
    "ButtonPress-v0",
    "CubeGrasp-v0",
]
