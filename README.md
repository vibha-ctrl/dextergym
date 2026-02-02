# Dexterous Manipulation Benchmark 🤖🖐️

A novel benchmark for dexterous manipulation using a Robotiq-style 3-finger gripper, trained with PPO reinforcement learning.

## Tasks

| Task | Description | Challenge |
|------|-------------|-----------|
| **USBInsertion-v0** | Insert a USB stick into a port | Tight tolerance alignment |
| **CoinStack-v0** | Stack 3 coins into a tower | Precision + sequential |
| **KeyTurn-v0** | Insert key and turn to unlock | Two-phase manipulation |
| **LightSwitch-v0** | Flip switches to target pattern | Quick precise movements |
| **BottleCap-v0** | Unscrew a bottle cap | Grip + rotate coordination |

## Quick Start

### 1. Install Dependencies

```bash
cd dexterous_benchmark
pip install -r requirements.txt
```

### 2. Test Environments

```bash
# Verify all environments work
python test_envs.py

# Test with visualization
python test_envs.py --render
```

### 3. Train

```bash
# Train single task
python train.py --task USBInsertion-v0

# Train all tasks (takes ~6-10 hours)
python train.py --task all

# Train with custom settings
python train.py --task CoinStack-v0 --timesteps 1000000 --n_envs 4
```

### 4. Evaluate

```bash
# Evaluate with visualization
python evaluate.py --task USBInsertion-v0 --render

# Benchmark all tasks
python evaluate.py --task all --episodes 50
```

### 5. Monitor Training

```bash
tensorboard --logdir tb_logs/
```

## Project Structure

```
dexterous_benchmark/
├── assets/
│   └── scenes/           # MuJoCo XML scene files
│       ├── usb_insertion.xml
│       ├── coin_stack.xml
│       ├── key_turn.xml
│       ├── light_switch.xml
│       └── bottle_cap.xml
├── envs/
│   ├── __init__.py       # Environment registration
│   ├── base_env.py       # Base environment class
│   ├── usb_insertion_env.py
│   ├── coin_stack_env.py
│   ├── key_turn_env.py
│   ├── light_switch_env.py
│   └── bottle_cap_env.py
├── models/               # Saved models (created during training)
├── logs/                 # Evaluation logs
├── tb_logs/              # TensorBoard logs
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── test_envs.py          # Environment tests
├── requirements.txt
└── README.md
```

## Robot: Robotiq-Style 3-Finger Gripper

The gripper has 12 degrees of freedom:
- **6 DoF base**: x, y, z translation + roll, pitch, yaw rotation
- **6 DoF fingers**: 3 fingers × 2 joints each (proximal + distal)

## Usage in Code

```python
import gymnasium as gym
import envs  # Register environments

# Create environment
env = gym.make("USBInsertion-v0", render_mode="human")

# Standard Gymnasium loop
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Replace with your policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Training with Custom Settings

```python
from stable_baselines3 import PPO
import gymnasium as gym
import envs

env = gym.make("CoinStack-v0")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    device="cuda",  # Use GPU
    verbose=1,
)

model.learn(total_timesteps=500_000)
model.save("my_model")
```

## Expected Results

After training (~500K timesteps per task):

| Task | Success Rate | Training Time (RTX 3060) |
|------|--------------|--------------------------|
| USBInsertion-v0 | 60-80% | ~1-2 hours |
| CoinStack-v0 | 40-60% | ~2-3 hours |
| KeyTurn-v0 | 50-70% | ~2-3 hours |
| LightSwitch-v0 | 70-90% | ~1-2 hours |
| BottleCap-v0 | 50-70% | ~1-2 hours |

*Results vary based on random seeds and hyperparameters.*

## Customization

### Modify Reward Function

Edit the `_get_reward()` method in any environment file:

```python
# envs/usb_insertion_env.py
def _get_reward(self) -> float:
    # Your custom reward logic
    ...
```

### Add New Task

1. Create scene XML in `assets/scenes/`
2. Create environment class in `envs/`
3. Register in `envs/__init__.py`

## Hardware Requirements

- **Minimum**: 4GB VRAM GPU (reduced parallel envs)
- **Recommended**: 6GB+ VRAM GPU (RTX 3060 or better)
- **CPU**: Works but slower (~5-10x)

## Citation

If you use this benchmark in your research:

```bibtex
@misc{dexterous_benchmark,
  title={Dexterous Manipulation Benchmark},
  year={2026},
  url={https://github.com/yourusername/dexterous_benchmark}
}
```

## License

MIT License
