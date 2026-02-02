"""
Coin Stacking Environment
=========================

Task: Stack 3 coins into a tower on the target zone.

Challenges:
- Thin objects require precise grasping
- Stability during placement
- Sequential multi-object manipulation

Success: All 3 coins stacked at target, tower stable.
"""

import numpy as np
from pathlib import Path
from .base_env import BaseDexterousEnv


class CoinStackEnv(BaseDexterousEnv):
    """
    Coin Stacking Task
    
    The agent must:
    1. Pick up coins one at a time
    2. Stack them at the target location
    3. Keep the stack stable
    """
    
    def __init__(self, render_mode=None):
        # Task parameters (must be set BEFORE super().__init__ because _get_obs is called)
        self.stack_threshold_xy = 0.015  # XY alignment tolerance
        self.stack_threshold_z = 0.01   # Height tolerance per coin
        self.coin_height = 0.006        # Height of each coin
        self.target_pos = np.array([0.1, 0, 0.171])  # Target zone center
        self.coin_names = ["coin_1", "coin_2", "coin_3"]
        
        model_path = Path(__file__).parent.parent / "assets/scenes/coin_stack.xml"
        super().__init__(str(model_path), render_mode)
        
        self.max_episode_steps = 500
        
    def _get_obs(self) -> np.ndarray:
        """
        Observation includes:
        - Gripper joint positions (12)
        - Gripper joint velocities (12) 
        - Gripper base position (3)
        - Coin 1 position (3)
        - Coin 2 position (3)
        - Coin 3 position (3)
        - Target position (3)
        - Fingertip positions (9)
        - Coins stacked count (1)
        """
        # Gripper state
        gripper_qpos = self.data.qpos[:12].copy()
        gripper_qvel = self.data.qvel[:12].copy()
        gripper_pos = self._get_gripper_pos()
        
        # Coin positions
        coin1_pos = self._get_body_pos("coin_1")
        coin2_pos = self._get_body_pos("coin_2")
        coin3_pos = self._get_body_pos("coin_3")
        
        # Target
        target_pos = self._get_site_pos("stack_target")
        
        # Fingertips
        fingertips = self._get_fingertip_positions().flatten()
        
        # Stacked count
        stacked = np.array([float(self._count_stacked_coins())])
        
        return np.concatenate([
            gripper_qpos,        # 12
            gripper_qvel,        # 12
            gripper_pos,         # 3
            coin1_pos,           # 3
            coin2_pos,           # 3
            coin3_pos,           # 3
            target_pos,          # 3
            fingertips,          # 9
            stacked,             # 1
        ]).astype(np.float32)
    
    def _count_stacked_coins(self) -> int:
        """Count how many coins are properly stacked at target."""
        count = 0
        expected_heights = [
            self.target_pos[2] + self.coin_height * 0.5,  # First coin
            self.target_pos[2] + self.coin_height * 1.5,  # Second coin
            self.target_pos[2] + self.coin_height * 2.5,  # Third coin
        ]
        
        # Check each coin
        coin_positions = [self._get_body_pos(name) for name in self.coin_names]
        
        for i, (target_h, coin_pos) in enumerate(zip(expected_heights, coin_positions)):
            # Check XY alignment
            xy_dist = np.linalg.norm(coin_pos[:2] - self.target_pos[:2])
            # Check height
            z_error = abs(coin_pos[2] - target_h)
            
            if xy_dist < self.stack_threshold_xy and z_error < self.stack_threshold_z:
                count += 1
            else:
                break  # Stack must be built from bottom up
        
        return count
    
    def _get_reward(self) -> float:
        """
        Reward function:
        1. Bonus for each stacked coin
        2. Distance shaping to encourage picking and placing
        3. Stability bonus
        """
        reward = 0.0
        
        # Count stacked coins
        stacked = self._count_stacked_coins()
        reward += stacked * 20.0  # Big bonus for each stacked coin
        
        # Get gripper and coin positions
        gripper_pos = self._get_gripper_pos()
        coin_positions = [self._get_body_pos(name) for name in self.coin_names]
        
        # Find nearest unstacked coin
        unstacked_coins = []
        for i, pos in enumerate(coin_positions):
            xy_dist = np.linalg.norm(pos[:2] - self.target_pos[:2])
            if xy_dist > self.stack_threshold_xy or i >= stacked:
                unstacked_coins.append((i, pos))
        
        if unstacked_coins:
            # Reward for approaching nearest unstacked coin
            nearest_coin_pos = unstacked_coins[0][1]
            gripper_to_coin = self._distance(gripper_pos, nearest_coin_pos)
            reward -= gripper_to_coin * 2.0
            
            # If holding a coin (gripper near coin), reward moving to target
            if gripper_to_coin < 0.05:
                coin_to_target = np.linalg.norm(
                    nearest_coin_pos[:2] - self.target_pos[:2]
                )
                reward -= coin_to_target * 5.0
        
        # Full success bonus
        if self._is_success():
            reward += 100.0
        
        # Time penalty
        reward -= 0.01
        
        return reward
    
    def _is_success(self) -> bool:
        """Success if all 3 coins are stacked."""
        return self._count_stacked_coins() >= 3
    
    def _is_terminated(self) -> bool:
        """Terminate if any coin falls off table."""
        for name in self.coin_names:
            pos = self._get_body_pos(name)
            if pos[2] < 0.1:  # Below table
                return True
        return False
    
    def _reset_task(self):
        """Randomize coin initial positions."""
        # Scatter coins on table
        positions = [
            [-0.1, 0.05, 0.175],
            [-0.1, -0.05, 0.175],
            [-0.05, 0, 0.175],
        ]
        
        # Add randomization
        for i, name in enumerate(self.coin_names):
            body_id = self.model.body(name).id
            jnt_id = self.model.body_jntadr[body_id]
            qpos_addr = self.model.jnt_qposadr[jnt_id]
            
            pos = np.array(positions[i])
            pos[:2] += self.np_random.uniform(-0.02, 0.02, size=2)
            
            self.data.qpos[qpos_addr:qpos_addr+3] = pos
            self.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        
        # Reset gripper
        self.data.qpos[0:3] = [0, 0, 0.35]
        self.data.qpos[3:6] = [0, 0, 0]
        self.data.qpos[6:12] = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
