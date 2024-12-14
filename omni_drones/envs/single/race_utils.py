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
import torch
from pathlib import Path
from omni_drones.utils.torch import euler_to_quaternion
from omni_drones.utils.math import quaternion_to_rotation_matrix


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
    
    # Gate's forward direction should be consistent
    forward = rot_matrix[:, 0]  # Negate to ensure consistent direction
    
    # Arrow starts from gate center
    start_point = pos
    end_point = (
        pos[0] + forward[0] * arrow_length,
        pos[1] + forward[1] * arrow_length,
        pos[2] + forward[2] * arrow_length
    )
    
    # Draw main arrow line
    draw_interface.draw_lines(
        [start_point], 
        [end_point],
        [(0.0, 1.0, 0.0, 1.0)],  # Green
        [arrow_scale]  # Line width
    )
    
    # Calculate arrow head
    arrow_back = (
        end_point[0] - forward[0] * arrow_width,
        end_point[1] - forward[1] * arrow_width,
        end_point[2] - forward[2] * arrow_width
    )
    
    # Calculate right vector for arrow head
    right = torch.tensor([forward[1], -forward[0], 0])
    right = right / torch.norm(right)
    
    arrow_right = (
        arrow_back[0] + right[0] * arrow_width,
        arrow_back[1] + right[1] * arrow_width,
        arrow_back[2] + right[2] * arrow_width
    )
    arrow_left = (
        arrow_back[0] - right[0] * arrow_width,
        arrow_back[1] - right[1] * arrow_width,
        arrow_back[2] - right[2] * arrow_width
    )
    
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
       
class TrackLoader:
    def __init__(self, track_file: str):
        self.track_path = Path(track_file)
        self.config = self._load_track()
        
    def _load_track(self):
        with open(self.track_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @property
    def gate_scale(self):
        return self.config.get('gate_scale', 1.1)
    
    @staticmethod
    def load_gates(config_dict):
        gates = []
        for gate in config_dict['gates']:
            # Convert degrees to radians for orientation
            roll, pitch, yaw = [angle * torch.pi/180 for angle in gate['ori']]
            
            gates.append({
                'id': gate['id'],
                'pos': tuple(gate['pos']),
                'ori': euler_to_quaternion(torch.tensor([roll, pitch, yaw])),
                'visible': gate.get('visible', True)
            })
        return gates