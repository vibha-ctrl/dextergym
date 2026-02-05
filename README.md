# DexterGym

Shadow Hand dexterous manipulation environments for reinforcement learning.

## Tasks

| Task | Description | Success |
|------|-------------|---------|
| `ButtonPress-v0` | Press a red button | Button pressed >1cm |
| `CubeGrasp-v0` | Grasp and lift a cube | Cube lifted above 35cm |
| `BallRotate-v0` | Rotate a ball in-hand | Full rotation while held |
| `PenSpin-v0` | Spin a pen with fingers | Half rotation while held |
| `BlockStack-v0` | Stack red block on blue | Block placed on target |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### View environments
```bash
python view.py --task ButtonPress-v0
python view.py --task all  # View all tasks
```

### Train with PPO
```bash
python train.py --task ButtonPress-v0 --timesteps 1000000
```

### Evaluate trained model
```bash
python evaluate.py --task ButtonPress-v0
```

## Code Structure

```
dextergym/
├── assets/
│   └── scenes/shadow.xml           # Shadow Hand (includable)
│   ├── mujoco_menagerie/           # Hand meshes
│   └── scenes/                     # Task scenes
├── envs/
│   ├── base_env.py                 # Base environment
│   ├── button_press_env.py
│   ├── cube_grasp_env.py
│   ├── ball_rotate_env.py
│   ├── pen_spin_env.py
│   └── block_stack_env.py
├── view.py                         # Visualization
├── train.py                        # Training script
└── evaluate.py                     # Evaluation script
```

## Notes

- All task objects use **primitive collision shapes** (box, sphere, cylinder, capsule) with stiff contact parameters to prevent penetration
- Shadow Hand uses mesh visuals but primitive collision geometry
- Contact stiffness: `solref="0.001 1" solimp="0.99 0.99 0.001"`
