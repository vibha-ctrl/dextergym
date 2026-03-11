"""CubePush-v0: Push a cube into a target ring with the Shadow Hand."""

import numpy as np
import mujoco
from pathlib import Path
from envs.base_env import BaseDexterousEnv


class CubePushEnv(BaseDexterousEnv):
    """Push a cube from center into a yellow target ring on the left."""
    
    def __init__(self, render_mode=None):
        self.target_pos = np.array([-0.15, 0.0])
        self.prev_cube_to_target = None
        model_path = str(Path(__file__).parent.parent / "assets/scenes/cube_push.xml")
        super().__init__(model_path, render_mode)
        self.max_episode_steps = 500
    
    def _reset_task(self):
        # Base position — hand starts to the right of the cube
        self._set_joint_qpos("base_x", 0.08 + np.random.uniform(-0.01, 0.01))
        self._set_joint_qpos("base_y", np.random.uniform(-0.01, 0.01))
        self._set_joint_qpos("base_z", -0.145)  # lowered so fingertips reach cube height
        self._set_joint_qpos("base_roll", 0.0)
        self._set_joint_qpos("base_pitch", 0.0)
        self._set_joint_qpos("base_yaw", 0.0)
        
        # Randomize finger joints (indices 6-25)
        for i in range(6, 26):
            self.data.qpos[i] = np.random.uniform(-0.1, 0.1)
        
        # Need fingertip positions to place cube relative to middle finger
        mujoco.mj_forward(self.model, self.data)
        
        mf_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'mf_tip')
        mf_tip_x = self.data.site_xpos[mf_sid][0]
        
        # Place cube 2mm left of middle fingertip (0.025 = cube half-width)
        cube_x = mf_tip_x - 0.002 - 0.025
        cube_y = np.random.uniform(-0.01, 0.01)
        
        cube_qpos_start = self.model.jnt_qposadr[self.model.joint('cube_joint').id]
        self.data.qpos[cube_qpos_start:cube_qpos_start + 3] = [cube_x, cube_y, 0.025]
        self.data.qpos[cube_qpos_start + 3:cube_qpos_start + 7] = [1, 0, 0, 0]
        
        self.data.qvel[:] = 0.0
        # Initialize ctrl to current qpos so delta actions start from rest
        for i in range(26):
            self.data.ctrl[i] = self.data.qpos[i]
        
        self.prev_cube_to_target = np.linalg.norm(
            np.array([cube_x, cube_y]) - self.target_pos
        )
    
    def _get_cube_pos(self) -> np.ndarray:
        return self._get_body_pos("cube")
    
    def _is_cube_touched_by_hand(self) -> bool:
        """Check physics contact between any hand geom and the cube."""
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
        tip_positions = np.concatenate([pos for pos in tips.values()])
        
        return np.concatenate([
            self.data.qpos[:26].copy(),
            self.data.qvel[:26].copy(),
            cube_pos,
            palm_pos - cube_pos,
            tip_positions,
            cube_pos[:2] - self.target_pos,
        ]).astype(np.float32)
    
    def _get_reward(self) -> float:
        cube_pos = self._get_cube_pos()
        base_x = self._get_joint_qpos("base_x")
        target_x = self.target_pos[0]  # -0.15
        
        # Reward hand moving left toward the ring (capped at target)
        move_reward = -max(base_x, target_x) * 50.0
        
        # Penalize overshooting past the ring
        overshoot_penalty = 0.0
        if base_x < target_x:
            overshoot_penalty = (target_x - base_x) * 300.0
        
        # Bonus for maintaining contact with cube
        contact_reward = 10.0 if self._is_cube_touched_by_hand() else 0.0
        
        # Reward cube moving closer to target (delta-based)
        cube_to_target = np.linalg.norm(cube_pos[:2] - self.target_pos)
        progress = 0.0
        if self.prev_cube_to_target is not None:
            progress = (self.prev_cube_to_target - cube_to_target) * 500.0
        self.prev_cube_to_target = cube_to_target
        
        # Reward cube proximity to target (absolute)
        cube_proximity = -cube_to_target * 30.0
        
        # Penalize cube tipping over
        cube_height_penalty = max(0, cube_pos[2] - 0.03) * 200.0
        
        # Penalize cube drifting sideways out of the ring (ring radius=0.055, cube half=0.025)
        y_drift = abs(cube_pos[1] - self.target_pos[1])
        y_drift_penalty = max(0, y_drift - 0.03) * 100.0
        
        # Success bonus scales with remaining steps so early success beats reward farming
        success_bonus = 0.0
        if self._is_success():
            remaining = self.max_episode_steps - self.current_step
            success_bonus = remaining * 20.0
        
        return (move_reward - overshoot_penalty + contact_reward + progress
                + cube_proximity - cube_height_penalty - y_drift_penalty + success_bonus)
    
    def _is_terminated(self) -> bool:
        return self._is_success()
    
    def _is_success(self) -> bool:
        cube_pos = self._get_cube_pos()
        dist_xy = np.linalg.norm(cube_pos[:2] - self.target_pos)
        ring_radius = 0.055
        cube_half_diag = np.sqrt(0.025**2 + 0.025**2)
        return dist_xy < (ring_radius - cube_half_diag)
