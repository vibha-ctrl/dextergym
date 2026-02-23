"""Cube Push Task - Push a cube into a target ring with Shadow Hand."""

import numpy as np
import mujoco
from pathlib import Path
from envs.base_env import BaseDexterousEnv


class CubePushEnv(BaseDexterousEnv):
    """Push a cube from the middle into a yellow target ring on the left."""
    
    def __init__(self, render_mode=None):
        self.target_pos = np.array([-0.15, 0.0])  # Ring XY position
        self.prev_cube_to_target = None
        model_path = str(Path(__file__).parent.parent / "assets/scenes/cube_push.xml")
        super().__init__(model_path, render_mode)
        self.max_episode_steps = 500
    
    def _reset_task(self):
        # Hand position
        self._set_joint_qpos("base_x", 0.08 + np.random.uniform(-0.02, 0.02))
        self._set_joint_qpos("base_y", np.random.uniform(-0.02, 0.02))
        self._set_joint_qpos("base_z", -0.145)  # Lowered so fingertips are at cube height (z≈0.025)
        self._set_joint_qpos("base_roll", 0.0)
        self._set_joint_qpos("base_pitch", 0.0)
        self._set_joint_qpos("base_yaw", 0.0)
        
        # Fingers slightly open
        for i in range(6, 26):
            self.data.qpos[i] = np.random.uniform(-0.1, 0.1)
        
        # Forward pass to compute fingertip positions
        mujoco.mj_forward(self.model, self.data)
        
        # Place cube 2mm to the left of the middle fingertip
        mf_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'mf_tip')
        mf_tip_x = self.data.site_xpos[mf_sid][0]
        
        cube_x = mf_tip_x - 0.002 - 0.025  # 2mm gap + cube half-width
        cube_y = np.random.uniform(-0.02, 0.02)
        
        cube_qpos_start = self.model.jnt_qposadr[self.model.joint('cube_joint').id]
        self.data.qpos[cube_qpos_start:cube_qpos_start + 3] = [cube_x, cube_y, 0.025]
        self.data.qpos[cube_qpos_start + 3:cube_qpos_start + 7] = [1, 0, 0, 0]
        
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.ctrl[0] = self._get_joint_qpos("base_x")
        self.data.ctrl[1] = self._get_joint_qpos("base_y")
        self.data.ctrl[2] = self._get_joint_qpos("base_z")
        
        # Init progress tracking
        self.prev_cube_to_target = np.linalg.norm(
            np.array([cube_x, cube_y]) - self.target_pos
        )
    
    def _get_cube_pos(self) -> np.ndarray:
        return self._get_body_pos("cube")
    
    def _is_cube_touched_by_hand(self) -> bool:
        """Check actual physics contact between hand and cube."""
        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'cube_geom')
        excluded = {'floor', 'ring_outer', 'ring_inner', 'cube_geom'}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == cube_geom_id or c.geom2 == cube_geom_id:
                other = c.geom2 if c.geom1 == cube_geom_id else c.geom1
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other)
                if name not in excluded:
                    return True
        return False
    
    def _get_obs(self) -> np.ndarray:
        cube_pos = self._get_cube_pos()
        palm_pos = self._get_palm_pos()
        tips = self._get_fingertip_positions()
        tip_positions = np.concatenate([pos for pos in tips.values()])  # 5x3 = 15
        
        return np.concatenate([
            self.data.qpos[:26].copy(),       # Hand joints (26)
            self.data.qvel[:26].copy(),        # Hand velocities (26)
            cube_pos,                           # Cube position (3)
            palm_pos - cube_pos,                # Palm to cube (3)
            tip_positions,                      # Fingertip positions (15)
            cube_pos[:2] - self.target_pos,     # Cube to target XY (2)
        ]).astype(np.float32)
    
    def _get_reward(self) -> float:
        cube_pos = self._get_cube_pos()
        
        # 1) MOVE LEFT: dominant signal — just move the hand left
        base_x = self._get_joint_qpos("base_x")
        move_reward = -base_x * 50.0  # base_x=0.08 → -4.0, base_x=-0.15 → +7.5
        
        # 2) CONTACT: big bonus for touching cube (fingers start 2mm away,
        #    moving left naturally causes contact)
        contact_reward = 10.0 if self._is_cube_touched_by_hand() else 0.0
        
        # 3) PROGRESS: reward cube moving closer to target
        cube_to_target = np.linalg.norm(cube_pos[:2] - self.target_pos)
        progress = 0.0
        if self.prev_cube_to_target is not None:
            progress = (self.prev_cube_to_target - cube_to_target) * 500.0
        self.prev_cube_to_target = cube_to_target
        
        # 4) CUBE PROXIMITY: reward cube being close to target
        cube_proximity = -cube_to_target * 30.0  # closer to ring = less negative
        
        # 5) SUCCESS
        success_bonus = 100.0 if self._is_success() else 0.0
        
        return move_reward + contact_reward + progress + cube_proximity + success_bonus
    
    def _is_success(self) -> bool:
        cube_pos = self._get_cube_pos()
        dist_xy = np.linalg.norm(cube_pos[:2] - self.target_pos)
        ring_radius = 0.055
        cube_half_diag = np.sqrt(0.025**2 + 0.025**2)  # ~0.035
        return dist_xy < (ring_radius - cube_half_diag)  # ~0.02
