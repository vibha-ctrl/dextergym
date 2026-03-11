"""CubePickup-v0: Grasp and lift a cube to a target height with the Shadow Hand."""

import numpy as np
import mujoco
from pathlib import Path
from envs.base_env import BaseDexterousEnv


class CubePickupEnv(BaseDexterousEnv):
    """Grasp a cube from the table and lift it to 10cm."""
    
    def __init__(self, render_mode=None):
        self.target_height = 0.10
        self.prev_cube_height = None
        model_path = str(Path(__file__).parent.parent / "assets/scenes/cube_pickup.xml")
        super().__init__(model_path, render_mode)
        self.max_episode_steps = 500
    
    def _reset_task(self):
        # Start hand at cube level so fingers surround it
        self._set_joint_qpos("base_x", np.random.uniform(-0.005, 0.005))
        self._set_joint_qpos("base_y", np.random.uniform(-0.005, 0.005))
        self._set_joint_qpos("base_z", -0.08 + np.random.uniform(-0.005, 0.005))
        self._set_joint_qpos("base_roll", 0.0)
        self._set_joint_qpos("base_pitch", 0.0)
        self._set_joint_qpos("base_yaw", 0.0)
        
        # Pre-grasp pose: fingers at mid-range with slight randomization
        for i in range(6, 26):
            low = self.model.jnt_range[i, 0]
            high = self.model.jnt_range[i, 1]
            mid = (low + high) / 2.0
            self.data.qpos[i] = mid + np.random.uniform(-0.05, 0.05)
        
        # Place cube centered under the hand
        cube_x = np.random.uniform(-0.005, 0.005)
        cube_y = np.random.uniform(-0.005, 0.005)
        
        cube_qpos_start = self.model.jnt_qposadr[self.model.joint('cube_joint').id]
        self.data.qpos[cube_qpos_start:cube_qpos_start + 3] = [cube_x, cube_y, 0.025]
        self.data.qpos[cube_qpos_start + 3:cube_qpos_start + 7] = [1, 0, 0, 0]
        
        self.data.qvel[:] = 0.0
        # Initialize ctrl to current qpos so delta actions start from rest
        for i in range(26):
            self.data.ctrl[i] = self.data.qpos[i]
        
        self.prev_cube_height = 0.025
    
    def _get_cube_pos(self) -> np.ndarray:
        return self._get_body_pos("cube")
    
    def _is_cube_touched_by_hand(self) -> bool:
        """Check physics contact between any hand geom and the cube."""
        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'cube_geom')
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == cube_geom_id or c.geom2 == cube_geom_id:
                other = c.geom2 if c.geom1 == cube_geom_id else c.geom1
                if other != cube_geom_id and other != floor_geom_id:
                    return True
        return False
    
    def _count_finger_contacts(self) -> int:
        """Count distinct fingers touching the cube (0-5).
        
        Uses parent body names since collision geoms are unnamed in shadow.xml.
        """
        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'cube_geom')
        touching_fingers = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == cube_geom_id or c.geom2 == cube_geom_id:
                other = c.geom2 if c.geom1 == cube_geom_id else c.geom1
                body_id = self.model.geom_bodyid[other]
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if body_name:
                    for prefix in ['th', 'ff', 'mf', 'rf', 'lf']:
                        if prefix in body_name:
                            touching_fingers.add(prefix)
                            break
        return len(touching_fingers)
    
    def _get_obs(self) -> np.ndarray:
        cube_pos = self._get_cube_pos()
        palm_pos = self._get_palm_pos()
        tips = self._get_fingertip_positions()
        tip_positions = np.concatenate([pos for pos in tips.values()])
        
        return np.concatenate([
            self.data.qpos[:26].copy(),
            self.data.qvel[:26].copy(),
            cube_pos,
            palm_pos - cube_pos,
            tip_positions,
            [cube_pos[2] - self.target_height],
        ]).astype(np.float32)
    
    def _get_reward(self) -> float:
        cube_pos = self._get_cube_pos()
        palm_pos = self._get_palm_pos()
        cube_height = cube_pos[2]
        tips = self._get_fingertip_positions()
        
        # Reward palm staying close to cube
        palm_to_cube = np.linalg.norm(palm_pos - cube_pos)
        reach_reward = -palm_to_cube * 40.0
        
        # Reward fingertips closing in on the cube (within 5cm)
        finger_proximity = 0.0
        for tip_pos in tips.values():
            dist = np.linalg.norm(tip_pos - cube_pos)
            finger_proximity += max(0, 0.05 - dist) * 20.0
        
        # Reward per finger in contact
        n_fingers = self._count_finger_contacts()
        contact_reward = n_fingers * 2.0
        
        # Big reward for lifting while grasping — scales with height and finger count
        above_table = max(0, cube_height - 0.025)
        grasp_lift_reward = 0.0
        if n_fingers >= 2:
            grasp_lift_reward = above_table * n_fingers * 200.0
        
        # Reward upward progress (delta)
        lift_progress = 0.0
        if self.prev_cube_height is not None:
            lift_progress = (cube_height - self.prev_cube_height) * 1000.0
        self.prev_cube_height = cube_height
        
        # Penalize cube drifting too far from center
        xy_drift = np.linalg.norm(cube_pos[:2])
        xy_drift_penalty = max(0, xy_drift - 0.05) * 50.0
        
        # Penalize dropping the cube after it was lifted
        drop_penalty = 0.0
        if self.prev_cube_height is not None and self.prev_cube_height > 0.04 and cube_height < 0.03:
            drop_penalty = 50.0
        
        # Success bonus scales with remaining steps so early success beats farming
        success_bonus = 0.0
        if self._is_success():
            remaining = self.max_episode_steps - self.current_step
            success_bonus = remaining * 20.0
        
        return (reach_reward + finger_proximity + contact_reward + grasp_lift_reward
                + lift_progress - xy_drift_penalty - drop_penalty + success_bonus)
    
    def _is_terminated(self) -> bool:
        return self._is_success()
    
    def _is_success(self) -> bool:
        cube_pos = self._get_cube_pos()
        height_ok = cube_pos[2] >= self.target_height
        xy_ok = np.linalg.norm(cube_pos[:2]) < 0.08
        grasped = self._count_finger_contacts() >= 2
        return height_ok and xy_ok and grasped
