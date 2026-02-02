"""
Button Press Environment
========================

Task: Press a big red button on a table.

The gripper starts above the button and must push it down.
Simple and satisfying!

Success: Button pressed down past threshold.
"""

import numpy as np
from pathlib import Path
from .base_env import BaseDexterousEnv


class ButtonPressEnv(BaseDexterousEnv):
    """
    Button Press Task
    
    Press the red button on the table!
    - Button radius: 8cm
    - Gripper starts 20cm above with random XY offset (±8cm)
    - Must navigate down and press
    """
    
    def __init__(self, render_mode=None):
        self.press_threshold = -0.009  # Button must be pushed ~1/2 down
        
        model_path = Path(__file__).parent.parent / "assets/scenes/button_press.xml"
        super().__init__(str(model_path), render_mode)
        
        self.max_episode_steps = 200
        
    def _get_obs(self) -> np.ndarray:
        """Observation: gripper state + button state."""
        # Button joint is qpos[0], gripper is qpos[1:13]
        gripper_qpos = self.data.qpos[1:13].copy()
        gripper_qvel = self.data.qvel[1:13].copy()
        gripper_pos = self._get_gripper_pos()
        
        button_pos = np.array([self._get_joint_qpos("button_joint")])
        button_body = self._get_body_pos("button")  # Button body position
        fingertips = self._get_fingertip_positions().flatten()
        
        return np.concatenate([
            gripper_qpos,    # 12
            gripper_qvel,    # 12
            gripper_pos,     # 3
            button_pos,      # 1
            button_body,     # 3
            fingertips,      # 9
        ]).astype(np.float32)  # Total: 40
    
    def _get_reward(self) -> float:
        """Simple reward: get close to button + press it."""
        gripper_pos = self._get_gripper_pos()
        button_pos = self._get_body_pos("button")
        button_pos[2] += 0.025  # Top of button
        
        # 1. Distance reward (closer = better)
        dist = np.linalg.norm(gripper_pos - button_pos)
        dist_reward = max(0, 1.0 - dist) * 10.0
        
        # 2. Press reward
        button_press = self._get_joint_qpos("button_joint")
        press_progress = max(0, (0.005 - button_press) / 0.030)
        press_reward = press_progress * 50.0
        
        # 3. Success bonus
        success_bonus = 100.0 if self._is_success() else 0.0
        
        return dist_reward + press_reward + success_bonus
    
    def _is_success(self) -> bool:
        """Success if button pressed down enough."""
        return self._get_joint_qpos("button_joint") < self.press_threshold
    
    def _is_terminated(self) -> bool:
        return False
    
    def _reset_task(self):
        """Reset button and gripper with randomization."""
        # Reset button
        self.data.qpos[0] = 0.005  # Button joint - slightly up
        
        # Gripper 50cm above button with random XY offset
        self.data.qpos[1] = self.np_random.uniform(-0.08, 0.08)  # x random
        self.data.qpos[2] = self.np_random.uniform(-0.08, 0.08)  # y random
        self.data.qpos[3] = 0.0     # z = 0.85 + 0 = 0.85 (50cm above button at 0.345)
        self.data.qpos[4:7] = [0, 0, 0]  # no rotation
        self.data.qpos[7:13] = [0.5, 0.3, 0.5, 0.3, 0.5, 0.3]  # fingers partly open
