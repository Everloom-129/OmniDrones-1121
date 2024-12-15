# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
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


import torch
import yaml
import torch.distributions as D
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec,
    CompositeSpec,
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec
)

import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import omni.isaac.core.objects as objects
from omni.isaac.debug_draw import _debug_draw

import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion
from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView

from omni_drones.robots import ASSET_PATH

from .race_utils import quat_to_matrix,TrackLoader, draw_gate_arrow
from omni_drones.utils.torch import quaternion_to_rotation_matrix

class FlyThrough(IsaacEnv):
    r"""
    A basic control task where the agent must fly the UAV through the gate.
    This task is a bit tricky because the gate moves horizontally in random locations.

    ## Observation

    - `drone_state` (16 + num_rotors): The basic information of the drone (except its position),
      containing its rotation (in quaternion), velocities (linear and angular),
      heading and up vectors, and the current throttle.
    - `target_drone_rpos` (3): The target position relative to the drone.
    - `gate_vel` (6): The linear and angular velocities of the gate.
    - `gate_drone_rpos` (2 * 2 = 4): The position of the gate relative to the drone's position.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.

    ## Reward

    - `pos`: Reward for maintaining the final position of the payload around the target position.
    - `gate`: Reward computed from the distance to the plane and the center of the gate.
    - `up`: Reward for maintaining an upright orientation.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.

    The total reward is computed as follows:

    ```{math}
        r = [r_\text{pos} + (0.5 * r_\text{gate}) + (r_\text{pos} + 0.3) * (r_\text{up} + r_\text{spin}) + r_\text{effort}]
    ```

    ## Episode End

    The episode ends when the drone gets too close or too far to the ground, or when the drone goes too
    far away horizontally, or when the drone gets too far to the gate, or when the drone passes by the gate, or when the maximum episode length
    is reached.

    ## Config

    | Parameter               | Type  | Default       | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "Hummingbird" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `reset_on_collision`    | bool  | False         | Indicates if the episode should reset when the drone collides with an obstacle.                                                                                                                                                         |
    | `gate_moving_range`     | float | 1.0           | Moving range of the gate.                                                                                                                                                                                                               |
    | `gate_scale`            | float | 1.1           | Scale of the gate.                                                                                                                                                                                                                      |
    | `reward_distance_scale` | float | 1.0           | Scales the reward based on the distance between the payload and its target.                                                                                                                                                             |
    | `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.reset_on_collision = cfg.task.reset_on_collision
        self.gate_scale = cfg.task.gate_scale
       
        map_path = "/home/tonyw/Projects/RL_drone/OD_main/cfg/task/RacingTrack/ellipse.yaml"
        # map_path = "./split_s.yaml"
        with open(map_path, 'r') as f:
            gates_config_dict = yaml.safe_load(f)
        
        self.gates_config = TrackLoader.load_gates(gates_config_dict)

        super().__init__(cfg, headless)

        self.drone.initialize()

        self.gates = []
        self.gate_frames = [] # TODO, check frames collision
        for i, gate_cfg in enumerate(self.gates_config):
            if gate_cfg['visible']:
                gate = ArticulationView(
                    f"/World/envs/env_*/Gate_{i}",
                    reset_xform_properties=False,
                    shape=[self.num_envs, 1],
                )
                gate.initialize()
                self.gates.append(gate)
            
            if gate_cfg['visible']:
                gate_frame = RigidPrimView(
                    f"/World/envs/env_*/Gate_{i}/frame",
                    reset_xform_properties=False,
                    shape=[self.num_envs, 1],
                    track_contact_forces=self.reset_on_collision
                )
                gate_frame.initialize()
                self.gate_frames.append(gate_frame)

        self.target = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False,
        )
        self.target.initialize()

        self.init_vels = torch.zeros_like(self.drone.get_velocities())
        self.init_joint_pos = self.drone.get_joint_positions(True)
        self.init_joint_vels = torch.zeros_like(self.drone.get_joint_velocities())
        self.obstacle_pos = self.get_env_poses(self.gates[0].get_world_poses())[0] # not used, TODO add obstacle
        self.target_pos = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.crossed_plane = torch.zeros(self.num_envs, 1, device=self.device, dtype=bool)

        self.init_pos_dist = D.Uniform(
            torch.tensor([-2.5, -1.5, 1.5], device=self.device),
            torch.tensor([-2.0, 1.5, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([.2, .2, 0.], device=self.device) * torch.pi
        )



        self.alpha = 0.7

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            color=(0.4, 0.26, 0.13),  # Brown wood-like color
        )

        scale = torch.ones(3) * self.cfg.task.gate_scale # expand to [32, 1, 3]
        for i, gate_cfg in enumerate(self.gates_config):
            if gate_cfg['visible']:
                prim_utils.create_prim(
                    f"/World/envs/env_0/Gate_{i}",
                    usd_path=ASSET_PATH + "/usd/gate_fixed.usd",
                    translation=gate_cfg["pos"],
                    scale=scale,
                    orientation=gate_cfg["ori"]
                )

        self.drone.spawn(translations=[(-2., 0.0, 2.0)])

        target = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=(5.0, 0., 2.),
            radius=0.05,
            color=torch.tensor([1., 0., 0.]) # red
        )
        kit_utils.set_collision_properties(target.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(target.prim_path, disable_gravity=True)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]   # 23
        observation_dim = drone_state_dim + 6
        if self.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
        
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, observation_dim))
            }
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "drone_uprightness": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "success": BinaryDiscreteTensorSpec(1, dtype=bool),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        drone_pos = self.init_pos_dist.sample((*env_ids.shape, 1))
        drone_rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.drone.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.drone.set_joint_velocities(self.init_joint_vels[env_ids], env_ids)

        self.crossed_plane[env_ids] = False

        target_pos = torch.tensor(self.gates_config[0]["pos"], device=self.device)
        self.target_pos[env_ids] = target_pos
        self.target.set_world_poses(
            target_pos + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids
        )

        self.stats.exclude("success")[env_ids] = 0.
        self.stats["success"][env_ids] = False

        if self._should_render(0) and (env_ids == self.central_env_idx).any():
            self.draw.clear_lines()
            for gate_cfg in self.gates_config:
                if gate_cfg['visible']:
                    draw_gate_arrow(self.draw,gate_cfg)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.drone_up = self.drone_state[..., 16:19]
        
        # 1. Gate Waypoint reward
        # 1.1 get gate position
        self.gates_pos = []
        for gate in self.gates:
            gate_pos = self.get_env_poses(gate.get_world_poses())[0]  # shape: [32, 1, 3]
            self.gates_pos.append(gate_pos)
        
        # 1.2 get nearest gate position
        drone_pos = self.drone_state[..., :3]  # shape: [32, 1, 3]
        distances = []
        for gate_pos in self.gates_pos:
            dist = torch.norm(gate_pos - drone_pos, dim=-1)  # shape: [32, 1]
            distances.append(dist)
        distances = torch.stack(distances, dim=-1)  # shape: [32, 1, num_gates]
        nearest_gate_idx = torch.argmin(distances, dim=-1)  # shape: [32, 1]
        
        batch_indices = torch.arange(self.num_envs, device=self.device)
        
        gates_pos_stack = torch.stack(self.gates_pos, dim=1)  # shape: [32, num_gates, 1, 3]
        self.nearest_gate_pos = gates_pos_stack[batch_indices, nearest_gate_idx.squeeze(-1)]  # shape: [32, 1, 3]
        
        # 2. relative position
        # 2.1 target position
        self.target_drone_rpos = self.target_pos - drone_pos  # shape: [32, 1, 3]
        self.gate_drone_rpos = self.nearest_gate_pos - drone_pos  # shape: [32, 1, 3]

        obs = [
            self.drone_state[..., 3:],  # shape: [32, 1, 20]
            self.target_drone_rpos,     # shape: [32, 1, 3]
            torch.zeros_like(self.gate_drone_rpos),  # shape: [32, 1, 3], was gate vel
            self.gate_drone_rpos,       # shape: [32, 1, 3]
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        
        obs = torch.cat(obs, dim=-1)

        self.pos_error = torch.norm(self.target_drone_rpos, dim=-1)
        self.stats["pos_error"].mul_(self.alpha).add_((1-self.alpha) * self.pos_error)
        self.stats["drone_uprightness"].mul_(self.alpha).add_((1-self.alpha) * self.drone_up[..., 2])

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # previously only consider one gate, need to define multiple gates
        crossed_plane = self.drone.pos[..., 0] > 0.
        crossing_plane = (crossed_plane & (~self.crossed_plane))
        self.crossed_plane |= crossing_plane
        distance_to_gate_plane = 0. - self.drone.pos[..., 0]
        distance_to_gate_center = torch.abs(self.drone.pos[..., 1:] - self.nearest_gate_pos[..., 1:])
        through_gate = (distance_to_gate_center < 0.5).all(-1)

        reward_gate = torch.where(
            distance_to_gate_plane > 0.,
            (0.4 - distance_to_gate_center).sum(-1) * torch.exp(-distance_to_gate_plane),
            1.
        )

        # pose reward
        distance_to_target = torch.norm(self.target_drone_rpos, dim=-1)

        # reward_pos = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance_to_target))
        reward_pos = torch.exp(-self.reward_distance_scale * distance_to_target)
        # uprightness
        reward_up = 0.5 * torch.square((self.drone_up[..., 2] + 1) / 2)

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 0.5 / (1.0 + torch.square(spin))

        if self.reset_on_collision:
            collision = (
                self.gate_frames[0]
                .get_net_contact_forces()
                .any(-1)
                .any(-1, keepdim=True)
            )
            # collision_reward = collision.float()

            # self.stats["collision"].add_(collision_reward)
        assert reward_pos.shape == reward_up.shape == reward_spin.shape
        reward = (
            reward_pos
            + 0.5 * reward_gate
            + (reward_pos + 0.3) * (reward_up + reward_spin)
            + reward_effort
        ) # * (1 - collision_reward)

        misbehave = (
            (self.drone.pos[..., 2] < 0.2)
            | (self.drone.pos[..., 2] > 2.5)
            | (self.drone.pos[..., 1].abs() > 2.)
            | (distance_to_target > 6.)
        )
        hasnan = torch.isnan(self.drone_state).any(-1)
        invalid = (crossing_plane & ~through_gate)

        terminated = misbehave | hasnan | invalid
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        if self.reset_on_collision:
            terminated |= collision

        self.stats["success"].bitwise_or_(distance_to_target < 0.2)
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
