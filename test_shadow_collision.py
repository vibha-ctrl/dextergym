#!/usr/bin/env python3
"""Test Shadow Hand collision with a solid object."""

import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("assets/scenes/shadow_test.xml")
data = mujoco.MjData(model)

print("=" * 50)
print("Shadow Hand Collision Test")
print("=" * 50)
print(f"Actuators: {model.nu}")
print(f"Joints: {model.njnt}")
print("\nWatching if hand penetrates the red box...")
print("Press Ctrl+C to stop\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 150
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.8
    viewer.cam.lookat[:] = [0.2, 0, 0.15]
    
    step = 0
    try:
        while viewer.is_running():
            if step % 50 == 0:
                for i in range(model.nu):
                    low = model.actuator_ctrlrange[i, 0]
                    high = model.actuator_ctrlrange[i, 1]
                    data.ctrl[i] = np.random.uniform(low, high)
            
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopped.")

print("Done!")
