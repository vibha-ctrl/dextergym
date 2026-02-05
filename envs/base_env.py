"""
Base Environment for Shadow Hand Manipulation Tasks
====================================================

Provides common functionality for all tasks:
- MuJoCo model loading with Shadow Hand
- Observation/action space definition
- Rendering
- Physics stepping
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    import mujoco.viewer as mj_viewer
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False
    mj_viewer = None


class BaseDexterousEnv(gym.Env):
    """
    Base class for Shadow Hand manipulation environments.
    
    Shadow Hand has:
    - 6 DoF floating base (x, y, z, roll, pitch, yaw)
    - 20 DoF hand joints (wrist + fingers)
    - Total: 26 actuators
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        model_path: str,
        render_mode: Optional[str] = None,
        frame_skip: int = 5,
    ):
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Simulation parameters
        self.dt = self.model.opt.timestep * frame_skip
        
        # Get actuator control ranges
        self.ctrl_range = self.model.actuator_ctrlrange.copy()
        
        # Action space: normalized [-1, 1] for all actuators
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )
        
        # Observation space (defined by subclass via _get_obs)
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32
        )
        
        # Rendering
        self.viewer = None
        self.renderer = None
        
        # Episode tracking
        self.current_step = 0
        self.max_episode_steps = 200
        
    def _get_obs(self) -> np.ndarray:
        """Get observation: joint positions + velocities."""
        return np.concatenate([
            self.data.qpos.flat.copy(),
            self.data.qvel.flat.copy(),
        ]).astype(np.float32)
    
    def _get_reward(self) -> float:
        """Compute reward. Must be implemented by subclass."""
        raise NotImplementedError
    
    def _is_success(self) -> bool:
        """Check if task is complete. Must be implemented by subclass."""
        raise NotImplementedError
    
    def _is_terminated(self) -> bool:
        """Check for early termination."""
        return False
    
    def _is_truncated(self) -> bool:
        """Check if episode should be truncated (time limit)."""
        return self.current_step >= self.max_episode_steps
    
    def _reset_task(self):
        """Reset task-specific state. Override in subclass."""
        pass
    
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale action from [-1, 1] to actuator control range."""
        low = self.ctrl_range[:, 0]
        high = self.ctrl_range[:, 1]
        scaled = low + (action + 1.0) * 0.5 * (high - low)
        return np.clip(scaled, low, high)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0
        self._reset_task()
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        info = {"is_success": self._is_success()}
        
        # Don't render here - let the first step handle it
        # This prevents the "jump" on reset
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        action = np.clip(action, -1.0, 1.0)
        ctrl = self._scale_action(action)
        
        self.data.ctrl[:] = ctrl
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
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            if not HAS_VIEWER:
                raise ImportError("MuJoCo viewer not available.")
            if self.viewer is None:
                self.viewer = mj_viewer.launch_passive(self.model, self.data)
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -25
                self.viewer.cam.distance = 0.8
                self.viewer.cam.lookat[:] = [0, 0, 0.3]
            self.viewer.sync()
            
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, 480, 640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer = None
    
    # Helper methods for Shadow Hand
    def _get_body_pos(self, name: str) -> np.ndarray:
        """Get position of a body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[body_id].copy()
    
    def _get_site_pos(self, name: str) -> np.ndarray:
        """Get position of a site."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_xpos[site_id].copy()
    
    def _get_joint_qpos(self, name: str) -> float:
        """Get position of a joint."""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_addr = self.model.jnt_qposadr[joint_id]
        return self.data.qpos[qpos_addr]
    
    def _set_joint_qpos(self, name: str, value: float):
        """Set position of a joint."""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_addr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_addr] = value
    
    def _get_palm_pos(self) -> np.ndarray:
        """Get palm position."""
        return self._get_site_pos("palm_site")
    
    def _get_fingertip_positions(self) -> Dict[str, np.ndarray]:
        """Get positions of all five fingertips."""
        return {
            "thumb": self._get_site_pos("th_tip"),
            "index": self._get_site_pos("ff_tip"),
            "middle": self._get_site_pos("mf_tip"),
            "ring": self._get_site_pos("rf_tip"),
            "little": self._get_site_pos("lf_tip"),
        }
    
    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two points."""
        return np.linalg.norm(a - b)
