"""ButtonPress-v0: Press a red button with the Shadow Hand."""

import numpy as np
from pathlib import Path
from envs.base_env import BaseDexterousEnv


class ButtonPressEnv(BaseDexterousEnv):
    """Press the red button down at least 4mm to succeed."""
    
    def __init__(self, render_mode=None):
        model_path = str(Path(__file__).parent.parent / "assets/scenes/button_press.xml")
        super().__init__(model_path, render_mode)
        self.max_episode_steps = 200
        self.success_threshold = 0.004
    
    def _reset_task(self):
        self._set_joint_qpos("button_joint", 0.0)
        
        # Randomize base position
        self._set_joint_qpos("base_x", np.random.uniform(-0.05, 0.05))
        self._set_joint_qpos("base_y", np.random.uniform(-0.05, 0.05))
        self._set_joint_qpos("base_z", np.random.uniform(-0.05, 0.05))
        self._set_joint_qpos("base_roll", 0.0)
        self._set_joint_qpos("base_pitch", 0.0)
        self._set_joint_qpos("base_yaw", 0.0)
        
        # Randomize finger joints (indices 6-25)
        for i in range(6, 26):
            self.data.qpos[i] = np.random.uniform(-0.2, 0.2)
        
        self.data.qvel[:] = 0.0
        # Initialize ctrl to current qpos so delta actions start from rest
        for i in range(26):
            self.data.ctrl[i] = self.data.qpos[i]
    
    def _get_obs(self) -> np.ndarray:
        button_pos = self._get_joint_qpos("button_joint")
        palm_pos = self._get_palm_pos()
        button_site = self._get_site_pos("button_site")
        
        return np.concatenate([
            self.data.qpos[:26].copy(),
            self.data.qvel[:26].copy(),
            [button_pos],
            palm_pos - button_site,
        ]).astype(np.float32)
    
    def _get_reward(self) -> float:
        button_pos = self._get_joint_qpos("button_joint")
        palm_pos = self._get_palm_pos()
        button_site = self._get_site_pos("button_site")
        
        dist_reward = -self._distance(palm_pos, button_site) * 2.0
        press_reward = -button_pos * 100.0  # button goes negative when pressed
        success_bonus = 10.0 if self._is_success() else 0.0
        
        return dist_reward + press_reward + success_bonus
    
    def _is_terminated(self) -> bool:
        return self._is_success()
    
    def _is_success(self) -> bool:
        return self._get_joint_qpos("button_joint") < -self.success_threshold
