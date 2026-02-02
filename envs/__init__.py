"""
Dexterous Benchmark Environments
================================

A collection of novel dexterous manipulation tasks using a Robotiq-style 3-finger gripper.

Tasks:
    - USBInsertion-v0: Insert a USB stick into a port
    - CoinStack-v0: Stack coins into a tower
    - KeyTurn-v0: Insert and turn a key in a lock
    - ButtonPress-v0: Press a big red button
    - BottleCap-v0: Twist off a bottle cap

Usage:
    import gymnasium as gym
    import envs  # Registers environments
    
    env = gym.make("USBInsertion-v0", render_mode="human")
"""

from gymnasium.envs.registration import register

# USB Insertion Task
register(
    id="USBInsertion-v0",
    entry_point="envs.usb_insertion_env:USBInsertionEnv",
    max_episode_steps=300,
)

# Coin Stacking Task
register(
    id="CoinStack-v0",
    entry_point="envs.coin_stack_env:CoinStackEnv",
    max_episode_steps=500,
)

# Key Turn Task
register(
    id="KeyTurn-v0",
    entry_point="envs.key_turn_env:KeyTurnEnv",
    max_episode_steps=400,
)

# Button Press Task
register(
    id="ButtonPress-v0",
    entry_point="envs.button_press_env:ButtonPressEnv",
    max_episode_steps=200,
)

# Bottle Cap Task
register(
    id="BottleCap-v0",
    entry_point="envs.bottle_cap_env:BottleCapEnv",
    max_episode_steps=300,
)

# List of all available tasks
TASKS = [
    "USBInsertion-v0",
    "CoinStack-v0", 
    "KeyTurn-v0",
    "ButtonPress-v0",
    "BottleCap-v0",
]
