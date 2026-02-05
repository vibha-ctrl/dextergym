"""Button Press Task - Press a red button with Shadow Hand."""

import numpy as np
from pathlib import Path
from envs.base_env import BaseDexterousEnv


class ButtonPressEnv(BaseDexterousEnv):
    """Press the red button down at least 1cm."""
    
    def __init__(self, render_mode=None):
        model_path = str(Path(__file__).parent.parent / "assets/scenes/button_press.xml")
        super().__init__(model_path, render_mode)
        self.max_episode_steps = 200
        self.success_threshold = 0.01  # Button pressed 1cm
    
    def _reset_task(self):
        # Reset button to unpressed
        self._set_joint_qpos("button_joint", 0.0)
        # Base position is now set in XML (0.35, 0, 0.35), so qpos=0 is correct
        self._set_joint_qpos("base_x", 0.0)
        self._set_joint_qpos("base_y", 0.0)
        self._set_joint_qpos("base_z", 0.0)
        self._set_joint_qpos("base_roll", 0.0)
        self._set_joint_qpos("base_pitch", 0.0)
        self._set_joint_qpos("base_yaw", 0.0)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0  # All controls to zero
    
    def _get_obs(self) -> np.ndarray:
        # Hand state + button state
        button_pos = self._get_joint_qpos("button_joint")
        palm_pos = self._get_palm_pos()
        button_site = self._get_site_pos("button_site")
        
        return np.concatenate([
            self.data.qpos[:26].copy(),  # Hand joints
            self.data.qvel[:26].copy(),  # Hand velocities
            [button_pos],                 # Button position
            palm_pos - button_site,       # Relative position
        ]).astype(np.float32)
    
    def _get_reward(self) -> float:
        button_pos = self._get_joint_qpos("button_joint")
        palm_pos = self._get_palm_pos()
        button_site = self._get_site_pos("button_site")
        
        # Distance reward: encourage getting close to button
        dist = self._distance(palm_pos, button_site)
        dist_reward = -dist * 2.0
        
        # Press reward: encourage pressing button down (negative = pressed)
        press_reward = -button_pos * 100.0  # Button goes negative when pressed
        
        # Success bonus
        success_bonus = 10.0 if self._is_success() else 0.0
        
        return dist_reward + press_reward + success_bonus
    
    def _is_success(self) -> bool:
        button_pos = self._get_joint_qpos("button_joint")
        return button_pos < -self.success_threshold
