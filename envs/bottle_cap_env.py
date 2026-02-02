"""
Bottle Cap Environment
======================

Task: Unscrew a bottle cap by rotating it.

Challenges:
- Tripod grip required
- Maintaining grip while rotating
- Coordinated finger and wrist movement

Success: Cap rotated enough to come off (rises up).
"""

import numpy as np
from pathlib import Path
from .base_env import BaseDexterousEnv


class BottleCapEnv(BaseDexterousEnv):
    """
    Bottle Cap Task
    
    The agent must grip the cap and rotate it counter-clockwise
    to unscrew it from the bottle.
    """
    
    def __init__(self, render_mode=None):
        model_path = Path(__file__).parent.parent / "assets/scenes/bottle_cap.xml"
        super().__init__(str(model_path), render_mode)
        
        # Task parameters
        self.rotation_target = 6.0      # ~340 degrees to fully unscrew (more realistic)
        self.lift_threshold = 0.025     # Cap must rise this much
        self.grip_threshold = 0.05      # Distance for cap to be "gripped"
        self.max_episode_steps = 300
        
    def _get_obs(self) -> np.ndarray:
        """
        Observation includes:
        - Gripper joint positions (12)
        - Gripper joint velocities (12)
        - Gripper base position (3)
        - Cap position (3)
        - Cap rotation angle (1)
        - Cap lift amount (1)
        - Bottle top position (3)
        - Fingertip positions (9)
        - Gripper to cap distance (1)
        """
        # Gripper state
        gripper_qpos = self.data.qpos[:12].copy()
        gripper_qvel = self.data.qvel[:12].copy()
        gripper_pos = self._get_gripper_pos()
        
        # Cap state
        cap_pos = self._get_body_pos("cap")
        cap_rotation = np.array([self._get_joint_qpos("cap_rotation")])
        cap_lift = np.array([self._get_joint_qpos("cap_lift")])
        
        # Bottle reference
        bottle_top = self._get_site_pos("bottle_top")
        
        # Fingertips
        fingertips = self._get_fingertip_positions().flatten()
        
        # Gripper to cap distance
        cap_site = self._get_site_pos("cap_site")
        gripper_to_cap = np.array([self._distance(gripper_pos, cap_site)])
        
        return np.concatenate([
            gripper_qpos,        # 12
            gripper_qvel,        # 12
            gripper_pos,         # 3
            cap_pos,             # 3
            cap_rotation,        # 1
            cap_lift,            # 1
            bottle_top,          # 3
            fingertips,          # 9
            gripper_to_cap,      # 1
        ]).astype(np.float32)
    
    def _is_gripping_cap(self) -> bool:
        """Check if gripper is close enough to cap."""
        gripper_pos = self._get_gripper_pos()
        cap_site = self._get_site_pos("cap_site")
        return self._distance(gripper_pos, cap_site) < self.grip_threshold
    
    def _get_reward(self) -> float:
        """
        Reward function:
        1. Approach cap
        2. Rotation progress
        3. Lift progress (cap coming off)
        4. Success bonus
        """
        reward = 0.0
        
        # Get state
        gripper_pos = self._get_gripper_pos()
        cap_site = self._get_site_pos("cap_site")
        cap_rotation = self._get_joint_qpos("cap_rotation")
        cap_lift = self._get_joint_qpos("cap_lift")
        
        # Distance to cap
        dist_to_cap = self._distance(gripper_pos, cap_site)
        
        # Phase 1: Approach cap
        reward -= dist_to_cap * 3.0
        
        # Phase 2: If gripping, reward rotation
        if self._is_gripping_cap():
            reward += 10.0  # Grip bonus
            
            # Rotation progress (normalize to [0, 1])
            rotation_progress = cap_rotation / self.rotation_target
            reward += rotation_progress * 30.0
            
            # Lift progress
            lift_progress = cap_lift / self.lift_threshold
            reward += lift_progress * 20.0
        
        # Success bonus
        if self._is_success():
            reward += 100.0
        
        # Time penalty
        reward -= 0.01
        
        return reward
    
    def _is_success(self) -> bool:
        """Success if cap is lifted off (unscrewed)."""
        cap_lift = self._get_joint_qpos("cap_lift")
        return cap_lift >= self.lift_threshold * 0.8
    
    def _reset_task(self):
        """Reset cap to closed position."""
        # Reset cap rotation and lift
        self._set_joint_qpos("cap_rotation", 0.0)
        self._set_joint_qpos("cap_lift", 0.0)
        
        # Reset gripper above bottle
        self.data.qpos[0:3] = [0, 0, 0.5]
        self.data.qpos[3:6] = [0, 0, 0]
        self.data.qpos[6:12] = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
