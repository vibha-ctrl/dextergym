"""
Key Turn Environment
====================

Task: Pick up a key, insert it into a lock, and turn it.

Challenges:
- Precise alignment for insertion
- Two-phase task (insert then rotate)
- Coordination of translation and rotation

Success: Key inserted and lock rotated 90 degrees.
"""

import numpy as np
from pathlib import Path
from .base_env import BaseDexterousEnv


class KeyTurnEnv(BaseDexterousEnv):
    """
    Key Turn Task
    
    The agent must:
    1. Grasp the key
    2. Insert key into lock
    3. Turn key 90 degrees (π/2 radians)
    """
    
    def __init__(self, render_mode=None):
        model_path = Path(__file__).parent.parent / "assets/scenes/key_turn.xml"
        super().__init__(str(model_path), render_mode)
        
        # Task parameters
        self.insertion_threshold = 0.02  # Distance for key considered inserted
        self.rotation_target = 1.57      # Target rotation (90 degrees)
        self.rotation_threshold = 0.15   # Tolerance for success
        self.max_episode_steps = 400
        
    def _get_obs(self) -> np.ndarray:
        """
        Observation includes:
        - Gripper joint positions (12)
        - Gripper joint velocities (12)
        - Gripper base position (3)
        - Key position (3)
        - Key orientation quaternion (4)
        - Key tip position (3)
        - Lock position (3)
        - Lock rotation angle (1)
        - Fingertip positions (9)
        """
        # Gripper state
        gripper_qpos = self.data.qpos[:12].copy()
        gripper_qvel = self.data.qvel[:12].copy()
        gripper_pos = self._get_gripper_pos()
        
        # Key state
        key_pos = self._get_body_pos("key")
        key_quat = self._get_body_quat("key")
        key_tip = self._get_site_pos("key_tip")
        
        # Lock state
        lock_pos = self._get_site_pos("lock_target")
        lock_rotation = np.array([self._get_joint_qpos("lock_rotation")])
        
        # Fingertips
        fingertips = self._get_fingertip_positions().flatten()
        
        return np.concatenate([
            gripper_qpos,        # 12
            gripper_qvel,        # 12
            gripper_pos,         # 3
            key_pos,             # 3
            key_quat,            # 4
            key_tip,             # 3
            lock_pos,            # 3
            lock_rotation,       # 1
            fingertips,          # 9
        ]).astype(np.float32)
    
    def _is_key_inserted(self) -> bool:
        """Check if key tip is close to lock."""
        key_tip = self._get_site_pos("key_tip")
        lock_pos = self._get_site_pos("lock_target")
        return self._distance(key_tip, lock_pos) < self.insertion_threshold
    
    def _get_reward(self) -> float:
        """
        Reward function:
        1. Approach and grasp key
        2. Insert key into lock
        3. Turn key
        """
        reward = 0.0
        
        # Get positions
        gripper_pos = self._get_gripper_pos()
        key_pos = self._get_body_pos("key")
        key_tip = self._get_site_pos("key_tip")
        lock_pos = self._get_site_pos("lock_target")
        lock_rotation = self._get_joint_qpos("lock_rotation")
        
        # Distances
        gripper_to_key = self._distance(gripper_pos, key_pos)
        key_to_lock = self._distance(key_tip, lock_pos)
        
        # Phase 1: Approach key
        reward -= gripper_to_key * 2.0
        
        # Phase 2: If holding key, approach lock
        if gripper_to_key < 0.08:
            reward -= key_to_lock * 5.0
        
        # Phase 3: If inserted, reward rotation
        if self._is_key_inserted():
            reward += 20.0  # Insertion bonus
            # Reward rotation progress
            rotation_progress = lock_rotation / self.rotation_target
            reward += rotation_progress * 30.0
        
        # Success bonus
        if self._is_success():
            reward += 100.0
        
        # Time penalty
        reward -= 0.01
        
        return reward
    
    def _is_success(self) -> bool:
        """Success if key is inserted and lock is rotated to target."""
        if not self._is_key_inserted():
            return False
        
        lock_rotation = self._get_joint_qpos("lock_rotation")
        return abs(lock_rotation - self.rotation_target) < self.rotation_threshold
    
    def _is_terminated(self) -> bool:
        """Terminate if key falls off table."""
        key_pos = self._get_body_pos("key")
        return key_pos[2] < 0.1
    
    def _reset_task(self):
        """Randomize key initial position."""
        # Random key position
        key_x = self.np_random.uniform(-0.12, -0.08)
        key_y = self.np_random.uniform(-0.03, 0.03)
        
        # Set key position
        key_body_id = self.model.body("key").id
        key_jnt_id = self.model.body_jntadr[key_body_id]
        qpos_addr = self.model.jnt_qposadr[key_jnt_id]
        
        self.data.qpos[qpos_addr:qpos_addr+3] = [key_x, key_y, 0.18]
        self.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        
        # Reset lock rotation
        self._set_joint_qpos("lock_rotation", 0.0)
        
        # Reset gripper
        self.data.qpos[0:3] = [0, 0, 0.35]
        self.data.qpos[3:6] = [0, 0, 0]
        self.data.qpos[6:12] = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
