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
        
        # Place cube 1 cm to the left of the leftmost fingertip
        tip_names = ['ff_tip', 'mf_tip', 'rf_tip', 'lf_tip', 'th_tip']
        tip_xs = []
        for name in tip_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            tip_xs.append(self.data.site_xpos[sid][0])
        leftmost_x = min(tip_xs)
        
        cube_x = leftmost_x - 0.01 - 0.025  # 1cm gap + cube half-width
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
        
        # 1) REACH: closest fingertip to cube side (XY only, ignore Z)
        cube_right_face_xy = np.array([cube_pos[0] + 0.025, cube_pos[1]])
        tips = self._get_fingertip_positions()
        closest_tip_dist = min(
            np.linalg.norm(tp[:2] - cube_right_face_xy) for tp in tips.values()
        )
        reach_reward = -closest_tip_dist * 20.0
        
        # 2) MOVE FORWARD: directly reward hand base_x being lower (= closer to cube/target)
        base_x = self._get_joint_qpos("base_x")
        forward_reward = -base_x * 5.0  # lower base_x = more reward
        
        # 3) PROGRESS: reward cube moving closer to target
        cube_to_target = np.linalg.norm(cube_pos[:2] - self.target_pos)
        progress = 0.0
        if self.prev_cube_to_target is not None:
            progress = (self.prev_cube_to_target - cube_to_target) * 300.0
        self.prev_cube_to_target = cube_to_target
        
        # 4) CONTACT: bonus for touching cube
        contact_reward = 3.0 if self._is_cube_touched_by_hand() else 0.0
        
        # 5) SUCCESS
        success_bonus = 100.0 if self._is_success() else 0.0
        
        # 6) Light penalties (small so they don't discourage movement)
        cube_height_penalty = -abs(cube_pos[2] - 0.025) * 3.0
        
        return (reach_reward + forward_reward + progress +
                contact_reward + success_bonus + cube_height_penalty)
    
    def _is_success(self) -> bool:
        cube_pos = self._get_cube_pos()
        dist_xy = np.linalg.norm(cube_pos[:2] - self.target_pos)
        ring_radius = 0.055
        cube_half_diag = np.sqrt(0.025**2 + 0.025**2)  # ~0.035
        return dist_xy < (ring_radius - cube_half_diag)  # ~0.02
