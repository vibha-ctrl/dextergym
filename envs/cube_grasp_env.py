"""Cube Grasp Task - Grasp and lift a cube with Shadow Hand."""

import numpy as np
from pathlib import Path
from envs.base_env import BaseDexterousEnv


class CubeGraspEnv(BaseDexterousEnv):
    """Grasp a cube and lift it above 10cm."""
    
    def __init__(self, render_mode=None):
        model_path = str(Path(__file__).parent.parent / "assets/scenes/cube_grasp.xml")
        super().__init__(model_path, render_mode)
        self.max_episode_steps = 200
        self.lift_threshold = 0.10  # Cube must be lifted 10cm above ground
        self.cube_initial_pos = np.array([0.0, 0.0, 0.025])
    
    def _reset_task(self):
        # Reset cube position with small random offset
        cube_x = 0.0 + np.random.uniform(-0.03, 0.03)
        cube_y = np.random.uniform(-0.03, 0.03)
        cube_z = 0.025  # Sitting on ground (half cube size)
        
        # Cube freejoint qpos: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        cube_qpos_start = self.model.jnt_qposadr[self.model.joint('cube_joint').id]
        self.data.qpos[cube_qpos_start:cube_qpos_start + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[cube_qpos_start + 3:cube_qpos_start + 7] = [1, 0, 0, 0]  # No rotation
        
        # Randomize hand starting position
        self._set_joint_qpos("base_x", np.random.uniform(-0.05, 0.05))
        self._set_joint_qpos("base_y", np.random.uniform(-0.05, 0.05))
        self._set_joint_qpos("base_z", 0.20 + np.random.uniform(-0.05, 0.05))
        self._set_joint_qpos("base_roll", 0.0)
        self._set_joint_qpos("base_pitch", 0.0)
        self._set_joint_qpos("base_yaw", 0.0)
        
        # Randomize finger joint angles
        for i in range(6, 26):
            self.data.qpos[i] = np.random.uniform(-0.2, 0.2)
        
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        # Set base controls to match starting joint positions so actuators hold in place
        self.data.ctrl[0] = self._get_joint_qpos("base_x")
        self.data.ctrl[1] = self._get_joint_qpos("base_y")
        self.data.ctrl[2] = self._get_joint_qpos("base_z")
    
    def _get_cube_pos(self) -> np.ndarray:
        """Get current cube position."""
        return self._get_body_pos("cube")
    
    def _get_obs(self) -> np.ndarray:
        cube_pos = self._get_cube_pos()
        palm_pos = self._get_palm_pos()
        
        return np.concatenate([
            self.data.qpos[:26].copy(),  # Hand joints
            self.data.qvel[:26].copy(),  # Hand velocities
            cube_pos,                     # Cube position (3)
            palm_pos - cube_pos,          # Palm to cube relative position (3)
        ]).astype(np.float32)
    
    def _get_reward(self) -> float:
        cube_pos = self._get_cube_pos()
        palm_pos = self._get_palm_pos()
        
        # Phase 1: Reach - get palm close to cube
        dist = self._distance(palm_pos, cube_pos)
        reach_reward = -dist * 5.0
        
        # Phase 2: Lift - reward cube height above ground
        cube_height = cube_pos[2]
        lift_reward = max(0, cube_height - 0.025) * 50.0  # Reward height above starting pos
        
        # Phase 3: Success bonus
        success_bonus = 20.0 if self._is_success() else 0.0
        
        # Penalty: cube falling off table (too far from start)
        cube_xy_dist = np.linalg.norm(cube_pos[:2] - self.cube_initial_pos[:2])
        drift_penalty = -cube_xy_dist * 2.0
        
        return reach_reward + lift_reward + success_bonus + drift_penalty
    
    def _is_success(self) -> bool:
        cube_pos = self._get_cube_pos()
        return cube_pos[2] > self.lift_threshold
