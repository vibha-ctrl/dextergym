#!/usr/bin/env python3
"""
Test if button physics works.
Commands the gripper to go straight down and checks if button compresses.
"""

import time
import gymnasium as gym
import numpy as np

# Register environments
import envs

print("=" * 50)
print("🔬 Physics Test: Can the gripper press the button?")
print("=" * 50)

env = gym.make("ButtonPress-v0", render_mode="human")
obs, info = env.reset()

print("\nCommanding gripper to GO DOWN...\n")

for step in range(300):
    # Action: all zeros except Z = go DOWN (negative = down in normalized action space)
    action = np.zeros(12)
    action[2] = -1.0  # Command Z to go to minimum (down!)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Get button state
    button_pos = env.unwrapped._get_joint_qpos("button_joint")
    gripper_z = env.unwrapped._get_gripper_pos()[2]
    
    if step % 20 == 0:
        print(f"Step {step:3d}: Gripper Z = {gripper_z:.3f}, Button = {button_pos:.4f}, Reward = {reward:.1f}")
        
        if button_pos < 0:
            print(f"  ✅ BUTTON IS BEING PRESSED!")
        if env.unwrapped._is_success():
            print(f"  🎉 SUCCESS THRESHOLD REACHED!")
    
    time.sleep(0.02)
    
    if terminated or truncated:
        break

# Final state
button_pos = env.unwrapped._get_joint_qpos("button_joint")
threshold = env.unwrapped.press_threshold
print("\n" + "=" * 50)
print(f"Final button position: {button_pos:.4f} (threshold: {threshold})")
if env.unwrapped._is_success():
    print("✅ SUCCESS! Physics works - button was pressed!")
elif button_pos < 0:
    print("⚠️  Button moved but not enough for success")
else:
    print("❌ Button didn't move - physics issue!")
print("=" * 50)

env.close()
