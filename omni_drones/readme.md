# OmniDrones Codebase

```bash
# Project Structure (Dec 06 2024)
# 44 directories, 121 files

omni_drones/
├── actuators/                  # Hardware abstraction layer for drone control
│   └── rotor_group.py         # Handles thrust calculations and motor control
│   └── __init__.py
│
├── controllers/               # Classical control algorithms
│   ├── cfg/                   # Controller configurations
│   │   ├── lee_controller_firefly.yaml
│   │   ├── lee_controller_hummingbird.yaml
│   │   └── lee_controller_neo11.yaml
│   ├── controller.py
│   └── lee_position_controller.py
│   └── __init__.py
│
├── envs/                     # Reinforcement learning environments
│   ├── dragon/               # Dragon-specific environments
│   │   ├── hover.py
│   │   └── __init__.py
│   ├── formation.py
│   ├── inv_pendulum/         # Inverted pendulum tasks
│   │   ├── inv_pendulum_fly_through.py
│   │   ├── inv_pendulum_hover.py
│   │   ├── inv_pendulum_track.py
│   │   ├── utils.py
│   │   └── __init__.py
│   ├── isaac_env.py
│   ├── payload/              # Payload manipulation tasks
│   │   ├── payload_fly_through.py
│   │   ├── payload_hover.py
│   │   ├── payload_track.py
│   │   ├── utils.py
│   │   └── __init__.py
│   ├── platform/             # Platform-based tasks
│   │   ├── platform_fly_through.py
│   │   ├── platform_hover.py
│   │   ├── platform_track.py
│   │   ├── utils.py
│   │   └── __init__.py
│   ├── rearrange.py
│   ├── single/               # Single drone tasks
│   │   ├── fly_through.py    # Navigation through waypoints
│   │   ├── forest.py         # Forest navigation
│   │   ├── hover.py          # Basic hovering
│   │   ├── track.py          # Target tracking
│   │   ├── pinball.py        # Pinball game environment
│   │   ├── track_devel.py
│   │   └── __init__.py
│   ├── transport/            # Object transportation tasks
│   │   ├── transport_fly_through.py
│   │   ├── transport_hover.py
│   │   ├── transport_track.py
│   │   ├── utils.py
│   │   └── __init__.py
│   ├── utils/                # Environment utilities
│   │   ├── helpers.py
│   │   ├── prims.py
│   │   ├── stage.py
│   │   └── __init__.py
│   └── __init__.py
│
├── learning/                 # Learning algorithms implementation
│   ├── common.py
│   ├── dqn.py               # Deep Q-Network
│   ├── happo.py             # Heterogeneous-Agent PPO
│   ├── mappo_new.py         # Updated Multi-Agent PPO
│   ├── mappo.py             # Multi-Agent PPO
│   ├── matd3.py             # Multi-Agent TD3
│   ├── modules/             # Neural network modules
│   │   ├── distributions.py
│   │   ├── networks.py
│   │   ├── rnn.py
│   │   └── __init__.py
│   ├── ppo/                 # PPO implementations
│   │   ├── common.py
│   │   ├── mappo.py
│   │   ├── ppo_adapt.py
│   │   ├── ppo.py
│   │   ├── ppo_rnn.py
│   │   └── __init__.py
│   ├── qmix.py              # QMIX algorithm
│   ├── sac.py               # Soft Actor-Critic
│   ├── td3.py               # Twin Delayed DDPG
│   ├── tdmpc.py             # TD Model Predictive Control
│   ├── utils/               # Learning utilities
│   │   ├── clip_grad.py
│   │   ├── gae.py
│   │   ├── valuenorm.py
│   │   └── __init__.py
│   └── __init__.py
│
├── robots/                   # Robot models and configurations
│   ├── assets/              # 3D models and configurations
│   │   └── usd/             # Universal Scene Description files
│   │       ├── cf2x_isaac.usd
│   │       ├── cf2x_pybullet.usd
│   │       ├── crazyflie.yaml
│   │       ├── dragon-4-2.usd
│   │       ├── dragon-4.usd
│   │       ├── dragon_link_0.usd
│   │       ├── dragon_link_1.usd
│   │       ├── dragon_link.usd
│   │       ├── firefly.usd
│   │       ├── firefly.yaml
│   │       ├── gate_fixed.usd
│   │       ├── gate_sliding.usd
│   │       ├── hummingbird.usd
│   │       ├── hummingbird.yaml
│   │       ├── iris.usd
│   │       ├── iris.yaml
│   │       ├── neo11.usd
│   │       ├── neo11.yaml
│   │       ├── omav.usd
│   │       ├── omav.yaml
│   │       ├── rotor.usd
│   │       └── transport_group.usd
│   ├── config.py            # Robot configurations
│   ├── drone/               # Drone model implementations
│   │   ├── crazyflie.py
│   │   ├── dragon.py
│   │   ├── firefly.py
│   │   ├── hummingbird.py
│   │   ├── iris.py
│   │   ├── multirotor.py    # Base multirotor class
│   │   ├── neo11.py
│   │   ├── omav.py
│   │   └── __init__.py
│   ├── robot.py             # Base robot class
│   └── __init__.py
│
├── sensors/                  # Sensor implementations
│   ├── camera.py
│   ├── config.py
│   └── __init__.py
│
├── utils/          
│   ├── bspline.py           # B-spline calculations
│   ├── envs/
│   │   └── __init__.py
│   ├── image.py             # Image processing
│   ├── kit.py               # Toolkit functions
│   ├── math.py              # Mathematical utilities
│   ├── poisson_disk.py      # Poisson disk sampling
│   ├── scene.py             # Scene management
│   ├── torch.py             # PyTorch utilities
│   ├── torchrl/             # TorchRL integration
│   │   ├── collector.py
│   │   ├── env.py
│   │   ├── transforms.py
│   │   └── __init__.py
│   ├── wandb.py
│   └── __init__.py
├── views
│   ├── __init__.py
└── __init__.py

## Simulator Structure 


### 1. High-Level Structure
The codebase is organized into several main components:

```bash
omni_drones/
├── actuators/      # Drone motor control
├── controllers/    # Classical control algorithms
├── envs/          # RL environments
├── learning/      # RL algorithms
├── robots/        # Drone models
├── sensors/       # Sensor implementations
└── utils/         # Helper functions
```

### 2. Key Components

#### Environments (`envs/`)
Contains different drone tasks:
- `single/`: Single drone tasks
  - `hover.py`: Basic hovering
  - `track.py`: Target tracking
  - `fly_through.py`: Navigation through waypoints
- `payload/`: Multi-drone payload transportation
- `platform/`: Landing tasks
- `transport/`: Transportation tasks

This is where you'd start if you want to understand the available tasks or create new ones.

#### Robots (`robots/`)
Different drone models and configurations:
- Popular drones like Crazyflie, Firefly, Hummingbird
- Base class `multirotor.py` for common drone functionality
- USD files in `assets/` define the physical properties and appearances

#### Learning Algorithms (`learning/`)
Implements various RL algorithms:
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)
- MAPPO (Multi-Agent PPO)
- QMIX (Q-mixing for multi-agent scenarios)
- TD3 (Twin Delayed DDPG)

#### Controllers (`controllers/`)
Classical control implementations:
- Lee position controller (common in drone control)
- Configuration files for different drone models

### 3. Getting Started

For a beginner, here's a suggested learning path:

1. **Start with Single Drone Tasks**:
   - Look at `envs/single/hover.py` for the simplest task
   - Understand how observations, actions, and rewards are defined

2. **Understanding the Environment**:
   - `isaac_env.py` is the base environment class
   - Built on Isaac Sim for physics simulation
   - Defines core functionality for all environments

3. **Training Process**:
   - Start with PPO algorithm (`learning/ppo/`)
   - Use the training script (`scripts/train.py`)
   - Configuration through YAML files

### 4. Key Features

1. **Modular Design**:
   - Easy to add new drone models
   - Flexible environment creation
   - Pluggable RL algorithms

2. **Realistic Simulation**:
   - Physics-based simulation using Isaac Sim
   - Accurate drone dynamics
   - Sensor modeling

3. **Multi-Agent Support**:
   - Multiple drones can interact
   - Collaborative tasks
   - Specialized multi-agent algorithms

### 5. Common Use Cases

1. **Single Drone Control**:
   ```python
   # Example task progression
   envs/single/hover.py  # Start here
   → envs/single/track.py  # Then try tracking
   → envs/single/fly_through.py  # Then navigation
   ```

2. **Multi-Drone Scenarios**:
   ```python
   envs/transport/  # Multiple drones carrying objects
   envs/formation/  # Formation flying
   ```

3. **Algorithm Selection**:
   - Single agent: Start with PPO or SAC
   - Multi-agent: Try MAPPO or QMIX

OmniDrones provides a comprehensive platform for drone RL research, from basic hovering to complex multi-agent scenarios, all built on the realistic physics of Isaac Sim + IssacLab.

Notice: 
- The codebase is under active development. 
- The structure and APIs are subject to change. 
- We are working on a more modular design to improve code reusability and extendability. 
- We welcome contributions and feedback from the community to help us improve OmniDrones. 
- Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for more information on how to contribute to the project. 