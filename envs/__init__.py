"""Shadow Hand dexterous manipulation environments."""

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
