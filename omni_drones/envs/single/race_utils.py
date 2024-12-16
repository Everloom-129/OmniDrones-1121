'''
  Racing Utility Function
  - track registration
  - preset few track trajectory

'''


# MIT License
#
# Copyright (c) 2024 Tony Wang, University of Pennsylvania
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.




import yaml
import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.distributions as D
from pathlib import Path
from omni_drones.utils.torch import euler_to_quaternion
from omni_drones.utils.math import quaternion_to_rotation_matrix
from scipy.interpolate import CubicSpline
import json

def ellipse(t, a=8, b=12, height=2.0):
    """Generate points along an ellipse trajectory"""
    x = a * torch.cos(t)
    y = b * torch.sin(t)
    z = torch.full_like(t, height) # same height for all points
    return torch.stack([x, y, z], dim=-1)

def quat_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def draw_gate_arrow(draw_interface, gate_cfg, arrow_scale=3):
    """Draw an arrow indicating gate direction."""
    arrow_length = 1.0
    arrow_width = 0.3
    pos = gate_cfg["pos"]
    ori = gate_cfg["ori"]

    # Convert quaternion to rotation matrix
    rot_matrix = quaternion_to_rotation_matrix(ori.unsqueeze(0))[0]  # [3, 3]

    # Gate's forward direction
    forward = rot_matrix[:, 0]  # [3]

    # Convert tensors to lists for drawing
    start_point = pos.tolist()  # Convert tensor to list
    end_point = [
        float(pos[0] + forward[0] * arrow_length),
        float(pos[1] + forward[1] * arrow_length),
        float(pos[2] + forward[2] * arrow_length)
    ]

    # Draw main arrow line
    draw_interface.draw_lines(
        [start_point],  # List of start points
        [end_point],    # List of end points
        [(0.0, 1.0, 0.0, 1.0)],  # Green color
        [arrow_scale]   # Line width
    )

    # Calculate arrow head points
    arrow_back = [
        float(end_point[0] - forward[0] * arrow_width),
        float(end_point[1] - forward[1] * arrow_width),
        float(end_point[2] - forward[2] * arrow_width)
    ]

    # Calculate right vector for arrow head
    right = torch.tensor([forward[1], -forward[0], 0])
    right = right / torch.norm(right)

    # Convert to lists for drawing
    arrow_right = [
        float(arrow_back[0] + right[0] * arrow_width),
        float(arrow_back[1] + right[1] * arrow_width),
        float(arrow_back[2] + right[2] * arrow_width)
    ]
    arrow_left = [
        float(arrow_back[0] - right[0] * arrow_width),
        float(arrow_back[1] - right[1] * arrow_width),
        float(arrow_back[2] - right[2] * arrow_width)
    ]

    # Draw arrow head
    draw_interface.draw_lines(
        [end_point],
        [arrow_right],
        [(0.0, 1.0, 0.0, 1.0)],
        [arrow_scale]
    )
    draw_interface.draw_lines(
        [end_point],
        [arrow_left],
        [(0.0, 1.0, 0.0, 1.0)],
        [arrow_scale]
    )

def debug_angle(gates_config):
    test_angles = [
        -torch.pi/4 - torch.pi/8,  # -5π/8
        -torch.pi/4 - torch.pi/16,  # -9π/16
        -torch.pi/4,               # -π/4
        -torch.pi/4 + torch.pi/16, # -7π/16
        -torch.pi/4 + torch.pi/8,  # -3π/8
    ]

    for i, angle in enumerate(test_angles):
        gates_config.append({
            "pos": (-6.3 - i*2, -7.4 - i*2, 2.),  # 每个门稍微错开一点位置
            "ori": euler_to_quaternion(torch.tensor([0., 0., angle]))
        })

        gates_config.append({
            "pos": (6.3 + i*2, 7.4 + i*2, 2.),  # 每个门稍微错开一点位置
            "ori": euler_to_quaternion(torch.tensor([0., 0., -angle]))
        })


def load_gates_from_yaml(config_dict, map_scale=1.0):
    """Load gates from config dictionary."""
    # Save gates config as JSON
    json_path = "tmp.json"
    with open(json_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

    gates = []
    for gate in config_dict['gates']:
        # Convert degrees to radians for orientation
            roll, pitch, yaw = [angle * torch.pi/180 for angle in gate['ori']]

            gates.append({
                'id': gate['id'],
                'pos': torch.tensor(gate['pos']) * map_scale, 
                'ori': euler_to_quaternion(torch.tensor([roll, pitch, yaw])),
                'visible': gate.get('visible', True)
            })
    return gates


def calculate_targets_config(gates_config):
    targets_config = []

    for gate in gates_config:
        # Extract the gate's position and orientation
        gate_pos = gate['pos']
        gate_ori = gate['ori']

        # Convert the quaternion orientation to a rotation matrix
        gate_rot_matrix = quat_to_matrix(gate_ori)

        # The local x-axis in the gate's frame is the first column of the rotation matrix
        local_x_axis = gate_rot_matrix[:, 0]

        # Calculate the target position 1 meter in front of the gate along the local x-axis
        target_pos = gate_pos + local_x_axis

        # Append the target position to the targets_config list
        targets_config.append({"pos": target_pos.tolist()})

    return targets_config


# Function to compute smooth trajectory
def generate_linear_trajectory(gates, num_points = 100):
    """Generate a direct linear trajectory through gates."""
    gate_positions = np.array([gate['pos'] for gate in gates])
    # Add gate 1 to the end for completing the loop
    gate_positions = np.vstack((gate_positions, gate_positions[0]))

    # Interpolating between gate positions
    trajectory = []
    for i in range(len(gate_positions) - 1):
        start = gate_positions[i]
        end = gate_positions[i + 1]
        segment = np.linspace(start, end, num_points)
        trajectory.append(segment)

    trajectory = np.vstack(trajectory)
    return trajectory

def generate_smooth_trajectory(gates, num_points=100):
    """Generate a smooth trajectory through gates using cubic splines."""
    # Convert gates to array and add first gate at end for loop closure
    gate_positions = np.array([gate['pos'] for gate in gates])
    gate_positions = np.vstack((gate_positions, gate_positions[0]))

    # Create parameter t (normalized distance along path)
    t = np.zeros(len(gate_positions))
    for i in range(1, len(gate_positions)):
        t[i] = t[i-1] + np.linalg.norm(gate_positions[i] - gate_positions[i-1])
    t = t / t[-1]  # Normalize to [0, 1]

    # Fit cubic splines for each dimension
    t_fine = np.linspace(0, 1, num_points)
    cs_x = CubicSpline(t, gate_positions[:, 0], bc_type='periodic')
    cs_y = CubicSpline(t, gate_positions[:, 1], bc_type='periodic')
    cs_z = CubicSpline(t, gate_positions[:, 2], bc_type='periodic')

    # Generate smooth trajectory
    trajectory = np.column_stack([
        cs_x(t_fine),
        cs_y(t_fine),
        cs_z(t_fine)
    ])

    return trajectory

if __name__ == "__main__":
    # Gate positions as per the UZH_track.yaml file
    gates = {
        1: np.array([-1.0, 1.25, 2.0]),
        2: np.array([8.0, 6.25, 2.0]),
        3: np.array([8.0, -3.125, 2.0]),
        4: np.array([-3.75, -6.0, 4.0]),
        5: np.array([-3.75, -6.0, 2.0]),
        6: np.array([4.375, -0.625, 2.0]),
        7: np.array([-2.5, 6.25, 2.0])
    }

    # Generate both trajectories
    linear_trajectory = generate_linear_trajectory(gates)
    smooth_trajectory = generate_smooth_trajectory(gates)

    # Plotting
    fig = plt.figure(figsize=(15, 5))

    # Plot linear trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    for gate_id, gate_pos in gates.items():
        ax1.scatter(*gate_pos, label=f"Gate {gate_id}", s=100)
    ax1.plot(linear_trajectory[:, 0], linear_trajectory[:, 1],
             linear_trajectory[:, 2], label="Linear", linewidth=2)
    ax1.set_title("Linear Trajectory")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    # Plot smooth trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    for gate_id, gate_pos in gates.items():
        ax2.scatter(*gate_pos, label=f"Gate {gate_id}", s=100)
    ax2.plot(smooth_trajectory[:, 0], smooth_trajectory[:, 1],
             smooth_trajectory[:, 2], label="Smooth", linewidth=2)
    ax2.set_title("Smooth Trajectory (Cubic Spline)")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("trajectory_comparison.png")
    plt.close()
