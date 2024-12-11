# Task Details
Jie Wang

12062024

## key differences between `Hover.yaml` and `HoverRand.yaml`:

1. **Environment Scale**:
   - `Hover`: Uses 128 environments with spacing of 6 units
   - `HoverRand`: Uses 4096 environments (much larger) with no specified spacing

2. **Main Distinguishing Feature**: `HoverRand` includes extensive randomization parameters that `Hover` doesn't have:

```yaml:cfg/task/HoverRand.yaml
# ... existing code ...
randomization:
  drone:
    train:
      mass_scale: [0.26, 1.74]          # Randomly varies drone mass
      inertia_scale: [0.026, 1.974]     # Randomly varies moment of inertia
      t2w_scale: [0.5556, 2.23]         # Thrust to weight ratio variation
      f2m_scale: [0.625, 2.5]           # Force to moment ratio variation
      drag_coef_scale: [0, 0.62]        # Drag coefficient variation
  payload:
    z: [-0.1, 0.1]                      # Payload vertical position variation
    mass: [0.01, 0.4]                   # Payload mass variation
# ... existing code ...
```

The key purpose difference is:
- `Hover`: Provides a basic hover task with fixed drone parameters
- `HoverRand`: Implements domain randomization for robust training, varying physical properties of both the drone and its payload to help the trained model generalize better to different conditions

This randomization makes `HoverRand` more suitable for training robust controllers that can handle varying physical parameters, while `Hover` is more focused on the basic hovering task with consistent parameters.



##  key aspects and insights of the Hover environment:

1. **Core Task Objective**
The environment is designed as a basic sanity check task where a drone needs to:
- Maintain a stable position in mid-air
- Keep a specific heading
- Avoid drifting
- Stay upright

2. **Observation Space Structure**
````python
# Observation components:
obs = [
    self.rpos,                    # Relative position to target (3)
    self.drone_state[..., 3:],   # Drone state excluding position (16 + num_rotors)
    self.rheading,               # Heading difference from reference (3)
    # Optional: time encoding (4) if self.time_encoding is True
]
````

3. **Reward Components**
The reward function is sophisticated and multiplicative:
````python
reward = (
    reward_pose +                                    # Position accuracy
    reward_pose * (reward_up + reward_spin) +       # Stability multiplier
    reward_effort +                                 # Energy efficiency
    reward_action_smoothness                        # Action smoothness
)
````
Key insights:
- The pose reward affects both position and stability rewards (multiplicative)
- Uses a smooth 1/(1+x²) function instead of exponential for rewards
- Includes energy efficiency through effort penalty
- Encourages smooth control through action smoothness

4. **Episode Termination Conditions**
````python
misbehave = (self.drone.pos[..., 2] < 0.2) | (distance > 4)  # Too low or too far
hasnan = torch.isnan(self.drone_state).any(-1)               # Invalid states
terminated = misbehave | hasnan
truncated = (self.progress_buf >= self.max_episode_length)
````

5. **Initialization and Randomization**
````python
# Position initialization ranges
self.init_pos_dist = D.Uniform(
    torch.tensor([-2.5, -2.5, 1.], device=self.device),    # Min bounds
    torch.tensor([2.5, 2.5, 2.5], device=self.device)      # Max bounds
)

# Rotation initialization ranges (in radians)
self.init_rpy_dist = D.Uniform(
    torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
    torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
)
````

6. **Optional Payload System**
- Can attach a payload to the drone using a prismatic joint
- Payload parameters can be randomized:
  - Z-position range
  - Mass range relative to drone mass
- Uses physics constraints for realistic behavior

7. **Performance Tracking**
````python
# Statistics tracked during training
stats_spec = CompositeSpec({
    "return": UnboundedContinuousTensorSpec(1),
    "episode_len": UnboundedContinuousTensorSpec(1),
    "pos_error": UnboundedContinuousTensorSpec(1),
    "heading_alignment": UnboundedContinuousTensorSpec(1),
    "uprightness": UnboundedContinuousTensorSpec(1),
    "action_smoothness": UnboundedContinuousTensorSpec(1),
})
````

8. **Design Choices**
- Uses exponential moving average (alpha=0.8) for stats tracking
- Implements vectorized environments for parallel training
- Uses TensorDict for structured data handling
- Supports both training and evaluation modes

9. **Configurable Parameters** (from YAML)
````yaml
reward_effort_weight: 0.1                # Energy efficiency importance
reward_action_smoothness_weight: 0.0     # Action smoothness importance
reward_distance_scale: 1.2               # Distance scaling in rewards
time_encoding: true                      # Whether to include time in observations
````

This environment serves as a good baseline for:
- Testing basic drone control algorithms
- Validating learning frameworks
- Building more complex drone environments
- Understanding fundamental drone stability challenges

The code is well-structured for both research and practical applications, with clear separation of concerns and extensive configurability.

##  key differences between the Track and Hover tasks:

1. **Task Objective**
- **Hover**: Maintain a fixed position and heading in mid-air
- **Track**: Follow a moving lemniscate (figure-8) trajectory in 3D space

2. **Reference Target**
- **Hover**:
````python
self.target_pos = torch.tensor([[0.0, 0.0, 2.]], device=self.device)  # Fixed position
````
- **Track**:
````python
# Dynamic target using lemniscate trajectory
self.target_pos = self._compute_traj(self.future_traj_steps)  # Moving target
````

3. **Observation Space**
- **Hover**:
````python
obs = [
    self.rpos,                    # Relative position (3)
    self.drone_state[..., 3:],   # Drone state (16 + num_rotors)
    self.rheading,               # Heading difference (3)
    # Optional time encoding (4)
]
````
- **Track**:
````python
obs = [
    self.rpos.flatten(1),        # Future trajectory positions (3 * future_traj_steps)
    self.drone_state[..., 3:],   # Drone state (16 + num_rotors)
    # Optional time encoding (4)
]
````

4. **Initialization**
- **Hover**: Random initial positions within a cube
````python
self.init_pos_dist = D.Uniform(
    torch.tensor([-2.5, -2.5, 1.], device=self.device),
    torch.tensor([2.5, 2.5, 2.5], device=self.device)
)
````
- **Track**: Initializes with trajectory parameters
````python
self.traj_c = self.traj_c_dist.sample()      # Trajectory center
self.traj_rot = euler_to_quaternion()         # Trajectory rotation
self.traj_scale = self.traj_scale_dist.sample() # Trajectory scale
self.traj_w = self.traj_w_dist.sample()      # Trajectory frequency
````

5. **Reset Conditions**
- **Hover**:
````python
misbehave = (self.drone.pos[..., 2] < 0.2) | (distance > 4)  # Height < 0.2 or distance > 4
````
- **Track**:
````python
misbehave = (self.drone.pos[..., 2] < 0.1) | (distance > self.reset_thres)  # Height < 0.1 or exceeds threshold
````

6. **Statistics Tracked**
- **Hover**:
````python
stats = {
    "pos_error": UnboundedContinuousTensorSpec(1),
    "heading_alignment": UnboundedContinuousTensorSpec(1),
    "uprightness": UnboundedContinuousTensorSpec(1),
    "action_smoothness": UnboundedContinuousTensorSpec(1),
}
````
- **Track**:
````python
stats = {
    "tracking_error": UnboundedContinuousTensorSpec(1),
    "tracking_error_ema": UnboundedContinuousTensorSpec(1),
    "action_smoothness": UnboundedContinuousTensorSpec(1),
}
````

7. **Visualization**
- **Hover**: Shows a static target position
- **Track**: Visualizes the entire lemniscate trajectory path using debug lines

8. **Complexity Level**
- **Hover**: Simpler task, good for initial testing and basic control
- **Track**: More complex task requiring prediction and dynamic trajectory following

9. **Payload Support**
- **Hover**: Has optional payload support with randomization
- **Track**: No built-in payload support

These differences make the Track task more challenging and suitable for testing advanced control strategies, while the Hover task serves as a good baseline for testing basic stability and control.