"""Cube Pickup Task - Grasp and lift a cube to a target height with Shadow Hand."""

import numpy as np
import mujoco
from pathlib import Path
from envs.base_env import BaseDexterousEnv


class CubePickupEnv(BaseDexterousEnv):
    """Grasp a cube from the table and lift it to a target height."""
    
    def __init__(self, render_mode=None):
        self.target_height = 0.10  # 10cm — must actually pick up the cube
        self.prev_cube_height = None
        model_path = str(Path(__file__).parent.parent / "assets/scenes/cube_pickup.xml")
        super().__init__(model_path, render_mode)
        self.max_episode_steps = 500
    
    def _reset_task(self):
        # Hand position: start RIGHT at cube level so fingers are around it
        self._set_joint_qpos("base_x", np.random.uniform(-0.005, 0.005))
        self._set_joint_qpos("base_y", np.random.uniform(-0.005, 0.005))
        self._set_joint_qpos("base_z", -0.08 + np.random.uniform(-0.005, 0.005))
        self._set_joint_qpos("base_roll", 0.0)
        self._set_joint_qpos("base_pitch", 0.0)
        self._set_joint_qpos("base_yaw", 0.0)
        
        # Fingers in pre-grasp: slightly curled inward (positive = curl)
        for i in range(6, 26):
            low = self.model.jnt_range[i, 0]
            high = self.model.jnt_range[i, 1]
            mid = (low + high) / 2.0
            self.data.qpos[i] = mid + np.random.uniform(-0.05, 0.05)
        
        # Place cube directly below the hand (minimal randomization)
        cube_x = np.random.uniform(-0.005, 0.005)
        cube_y = np.random.uniform(-0.005, 0.005)
        
        cube_qpos_start = self.model.jnt_qposadr[self.model.joint('cube_joint').id]
        self.data.qpos[cube_qpos_start:cube_qpos_start + 3] = [cube_x, cube_y, 0.025]
        self.data.qpos[cube_qpos_start + 3:cube_qpos_start + 7] = [1, 0, 0, 0]
        
        self.data.qvel[:] = 0.0
        # Set ALL controls to match starting joint positions
        for i in range(26):
            self.data.ctrl[i] = self.data.qpos[i]
        
        # Init progress tracking
        self.prev_cube_height = 0.025
    
    def step(self, action):
        """Override step to use DELTA actions: action=0 means 'hold position'."""
        action = np.clip(action, -1.0, 1.0)
        # Delta: adjust current ctrl by action * scale
        delta_scale = 0.05  # Each step can adjust ctrl by ±0.05
        low = self.ctrl_range[:, 0]
        high = self.ctrl_range[:, 1]
        self.data.ctrl[:] = np.clip(
            self.data.ctrl + action * delta_scale * (high - low),
            low, high
        )
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        self.current_step += 1
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = {"is_success": self._is_success()}
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def _get_cube_pos(self) -> np.ndarray:
        return self._get_body_pos("cube")
    
    def _is_cube_touched_by_hand(self) -> bool:
        """Check actual physics contact between hand and cube."""
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
        """Count how many distinct fingers are touching the cube."""
        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'cube_geom')
        touching_fingers = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == cube_geom_id or c.geom2 == cube_geom_id:
                other = c.geom2 if c.geom1 == cube_geom_id else c.geom1
                # Geom names are None (unnamed), so check BODY name instead
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
        tip_positions = np.concatenate([pos for pos in tips.values()])  # 5x3 = 15
        
        return np.concatenate([
            self.data.qpos[:26].copy(),             # Hand joints (26)
            self.data.qvel[:26].copy(),             # Hand velocities (26)
            cube_pos,                                # Cube position (3)
            palm_pos - cube_pos,                     # Palm to cube (3)
            tip_positions,                           # Fingertip positions (15)
            [cube_pos[2] - self.target_height],      # Cube height relative to target (1)
        ]).astype(np.float32)
    
    def _get_reward(self) -> float:
        cube_pos = self._get_cube_pos()
        palm_pos = self._get_palm_pos()
        cube_height = cube_pos[2]
        tips = self._get_fingertip_positions()
        
        # 1) REACH: reward palm staying close to cube in 3D
        palm_to_cube_3d = np.linalg.norm(palm_pos - cube_pos)
        reach_reward = -palm_to_cube_3d * 40.0
        
        # 2) FINGER CLOSURE: reward each fingertip for being close to cube
        #    This teaches the agent to curl fingers AROUND the cube
        finger_proximity = 0.0
        for tip_pos in tips.values():
            dist = np.linalg.norm(tip_pos - cube_pos)
            finger_proximity += max(0, 0.05 - dist) * 20.0  # reward within 5cm
        
        # 3) CONTACT: reward actual physics contact, more for multi-finger
        n_fingers = self._count_finger_contacts()
        contact_reward = n_fingers * 2.0
        
        # 4) GRASP + LIFT: the big payoff — only get this by lifting while grasping
        above_table = max(0, cube_height - 0.025)
        grasp_lift_reward = 0.0
        if n_fingers >= 2:
            grasp_lift_reward = above_table * 500.0  # Massive reward for lifting with grip
        
        # 5) LIFT PROGRESS: reward cube moving upward (delta)
        lift_progress = 0.0
        if self.prev_cube_height is not None:
            lift_progress = (cube_height - self.prev_cube_height) * 1000.0
        self.prev_cube_height = cube_height
        
        # 6) XY DRIFT PENALTY: moderate
        cube_xy_drift = np.linalg.norm(cube_pos[:2])
        xy_drift_penalty = max(0, cube_xy_drift - 0.05) * 50.0
        
        # 7) DROP PENALTY: if cube was lifted but fell back
        drop_penalty = 0.0
        if self.prev_cube_height is not None and self.prev_cube_height > 0.04 and cube_height < 0.03:
            drop_penalty = 50.0
        
        # 8) SUCCESS: bonus scales with remaining steps
        success_bonus = 0.0
        if self._is_success():
            remaining = self.max_episode_steps - self.current_step
            success_bonus = remaining * 20.0
        
        return reach_reward + finger_proximity + contact_reward + grasp_lift_reward + lift_progress - xy_drift_penalty - drop_penalty + success_bonus
    
    def _is_terminated(self) -> bool:
        """End episode early if cube is at target height."""
        return self._is_success()
    
    def _is_success(self) -> bool:
        cube_pos = self._get_cube_pos()
        # Cube must actually reach target height while being grasped
        height_ok = cube_pos[2] >= self.target_height  # Must reach full 5cm
        xy_ok = np.linalg.norm(cube_pos[:2]) < 0.08  # Not drifted too far
        grasped = self._count_finger_contacts() >= 2  # Must still be holding it
        return height_ok and xy_ok and grasped
