"""
USB Insertion Environment
=========================

Task: Pick up a USB stick and insert it into a USB port.

Challenges:
- Precise alignment required
- Tight tolerance insertion
- Orientation matters (USB-A has a "right way up")

Success: USB tip is inside the port slot.
"""

import numpy as np
from pathlib import Path
from .base_env import BaseDexterousEnv


class USBInsertionEnv(BaseDexterousEnv):
    """
    USB Insertion Task
    
    The agent must:
    1. Move gripper to USB stick
    2. Grasp the USB stick
    3. Move to align with port
    4. Insert USB into port
    """
    
    def __init__(self, render_mode=None):
        model_path = Path(__file__).parent.parent / "assets/scenes/usb_insertion.xml"
        super().__init__(str(model_path), render_mode)
        
        # Task parameters
        self.insertion_threshold = 0.015  # Distance for successful insertion
        self.grasp_threshold = 0.03  # Distance to consider USB grasped
        self.max_episode_steps = 300
        
        # Target position (USB port)
        self.port_pos = np.array([0.15, 0, 0.2])
        
    def _get_obs(self) -> np.ndarray:
        """
        Observation includes:
        - Gripper joint positions (12)
        - Gripper joint velocities (12)
        - Gripper base position (3)
        - USB position (3)
        - USB orientation quaternion (4)
        - USB tip position (3)
        - Port position (3)
        - Distance USB tip to port (1)
        - Fingertip positions (9)
        """
        # Gripper state
        gripper_qpos = self.data.qpos[:12].copy()
        gripper_qvel = self.data.qvel[:12].copy()
        gripper_pos = self._get_gripper_pos()
        
        # USB state
        usb_pos = self._get_body_pos("usb_stick")
        usb_quat = self._get_body_quat("usb_stick")
        usb_tip = self._get_site_pos("usb_tip")
        
        # Port state
        port_pos = self._get_site_pos("port_target")
        
        # Relative info
        tip_to_port_dist = np.array([self._distance(usb_tip, port_pos)])
        
        # Fingertips
        fingertips = self._get_fingertip_positions().flatten()
        
        return np.concatenate([
            gripper_qpos,        # 12
            gripper_qvel,        # 12
            gripper_pos,         # 3
            usb_pos,             # 3
            usb_quat,            # 4
            usb_tip,             # 3
            port_pos,            # 3
            tip_to_port_dist,    # 1
            fingertips,          # 9
        ]).astype(np.float32)
    
    def _get_reward(self) -> float:
        """
        Reward function:
        1. Distance-based shaping (gripper to USB, USB to port)
        2. Grasp bonus
        3. Insertion success bonus
        """
        # Get positions
        gripper_pos = self._get_gripper_pos()
        usb_pos = self._get_body_pos("usb_stick")
        usb_tip = self._get_site_pos("usb_tip")
        port_pos = self._get_site_pos("port_target")
        
        # Distances
        gripper_to_usb = self._distance(gripper_pos, usb_pos)
        tip_to_port = self._distance(usb_tip, port_pos)
        
        reward = 0.0
        
        # Phase 1: Approach USB
        approach_reward = -gripper_to_usb * 2.0
        reward += approach_reward
        
        # Phase 2: If close to USB, reward getting USB to port
        if gripper_to_usb < 0.1:
            alignment_reward = -tip_to_port * 5.0
            reward += alignment_reward
        
        # Phase 3: Insertion bonus
        if self._is_success():
            reward += 100.0
        
        # Small penalty for time
        reward -= 0.01
        
        return reward
    
    def _is_success(self) -> bool:
        """Success if USB tip is close to port target."""
        usb_tip = self._get_site_pos("usb_tip")
        port_pos = self._get_site_pos("port_target")
        return self._distance(usb_tip, port_pos) < self.insertion_threshold
    
    def _is_terminated(self) -> bool:
        """Terminate if USB falls off table."""
        usb_pos = self._get_body_pos("usb_stick")
        return usb_pos[2] < 0.1  # Below table
    
    def _reset_task(self):
        """Randomize USB initial position."""
        # Random USB position on table
        usb_x = self.np_random.uniform(-0.1, 0.0)
        usb_y = self.np_random.uniform(-0.05, 0.05)
        
        # Set USB position (freejoint: 3 pos + 4 quat)
        # Find the USB joint qpos address
        usb_body_id = self.model.body("usb_stick").id
        usb_jnt_id = self.model.body_jntadr[usb_body_id]
        qpos_addr = self.model.jnt_qposadr[usb_jnt_id]
        
        # Set position
        self.data.qpos[qpos_addr:qpos_addr+3] = [usb_x, usb_y, 0.19]
        # Set orientation (identity quaternion)
        self.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        
        # Reset gripper to initial position
        self.data.qpos[0:3] = [0, 0, 0.35]  # x, y, z
        self.data.qpos[3:6] = [0, 0, 0]     # roll, pitch, yaw
        self.data.qpos[6:12] = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]  # fingers slightly open
