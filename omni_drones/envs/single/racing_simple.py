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

import yaml
import torch
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

from .race_utils import *


class RacingSimple(IsaacEnv):
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
        self.gate_moving_range = cfg.task.gate_moving_range
        self.gate_scale = cfg.task.gate_scale
        self.trajectory_type = cfg.task.trajectory_type

        map_path = f"/home/tonyw/Projects/RL_drone/OD_main/cfg/task/RacingTrack/{self.trajectory_type}.yaml"
        with open(map_path, 'r') as f:
            gates_config_yaml = yaml.safe_load(f)
        self.gates_config = load_gates_from_yaml(gates_config_yaml)

        self.targets_config = calculate_targets_config(self.gates_config)

        super().__init__(cfg, headless)

        self.drone.initialize()

        self.gates = []
        self.gate_frames = []
        for i in range(len(self.gates_config)):
            gate = ArticulationView(
                f"/World/envs/env_*/Gate_{i}",
                reset_xform_properties=False,
                shape=[self.num_envs, 1],
            )
            gate.initialize()
            self.gates.append(gate)

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

        self.next_gate_idx = torch.zeros(self.num_envs, 1, device=self.device, dtype=int)

        self.init_pos_dist = D.Uniform(
            torch.tensor([7.5, 3.5, 1.5], device=self.device),
            torch.tensor([8.5, 4.5, 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([.2, .2, 0.], device=self.device) * torch.pi
        )
        self.init_gate_pos_dist = D.Uniform(
            torch.tensor([-self.gate_moving_range], device=self.device),
            torch.tensor([self.gate_moving_range], device=self.device)
        )
        # self.target_pos_dist = D.Uniform(
        #     torch.tensor([1.5, -1., 1.5], device=self.device),
        #     torch.tensor([2.5, 1., 2.5], device=self.device)
        # )

        # self.base_target_pos = torch.tensor([8.0, 0.0, 2.0], device=self.device)


        self.alpha = 0.7


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
        )

        scale = torch.ones(3) * self.cfg.task.gate_scale # expand to [32, 1, 3]
        for i, gate_cfg in enumerate(self.gates_config):
            prim_utils.create_prim(
                f"/World/envs/env_0/Gate_{i}",
                usd_path=ASSET_PATH + "/usd/gate_fixed.usd",
                translation=gate_cfg["pos"],
                scale=scale,
                orientation=gate_cfg["ori"]
            )

        # self.drone.spawn(translations=[(-2., 0.0, 2.0)])
        self.drone.spawn(translations=[(-8.0, -2.0, 2.0)])

        target = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=(1.5, 0., 2.),
            radius=0.05,
            color=torch.tensor([1., 0., 0.])
        )
        kit_utils.set_collision_properties(target.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(target.prim_path, disable_gravity=True)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3
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

        self.next_gate_idx[env_ids] = 0

        drone_state = self.drone.get_state()
        self.prev_drone_pos = drone_state[..., :3]
        target_pos = torch.tensor(self.targets_config[0]["pos"], device=self.device)
        self.target_pos[env_ids] = target_pos
        self.target.set_world_poses(
            target_pos + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids
        )

        self.stats.exclude("success")[env_ids] = 0.
        self.stats["success"][env_ids] = False

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self.drone_up = self.drone_state[..., 16:19]
        drone_pos = self.drone_state[..., :3]

        gates_pos = [config['pos'].tolist() for config in self.gates_config]
        gates_pos_tensor = torch.tensor(gates_pos, device=self.device)
        self.gates_pos = gates_pos_tensor[self.next_gate_idx.squeeze()].unsqueeze(1)

        gate_ori = [config['ori'].tolist() for config in self.gates_config]
        gate_ori_tensor = torch.tensor(gate_ori, device=self.device)
        self.gate_ori = gate_ori_tensor[self.next_gate_idx.squeeze()].unsqueeze(1)

        target_pos = [config['pos'] for config in self.targets_config]
        target_pos_tensor = torch.tensor(target_pos, device=self.device)
        self.target_pos = target_pos_tensor[self.next_gate_idx.squeeze()].unsqueeze(1)

        # self.target_pos = self.gates_pos
        self.gate_drone_rpos = self.gates_pos - drone_pos
        # self.gate_drone_rpos = self.gates_pos[0] - self.drone_state[..., :3]
        self.target_drone_rpos = self.target_pos - drone_pos

        # self.drone_pos_diff = drone_pos - self.prev_drone_pos

        # Convert gate orientation to rotation matrix
        gate_rot = quat_to_matrix(self.gate_ori)  # [32, 1, 3, 3]
        # Get gate's forward direction (assuming gate's forward is along local X axis)
        gate_forward = gate_rot[..., 0]  # [32, 1, 3]
        gate_to_drone = drone_pos - self.gates_pos  # [32, 1, 3]
        # self.distance_to_gate_center = torch.norm(gate_to_drone, dim=-1)

        self.prev_target_drone_rpos = self.target_pos - self.prev_drone_pos

        prev_next_gate_drone_rpos = self.prev_drone_pos - self.gates_pos
        gate_to_drone = drone_pos - self.gates_pos  # [32, 1, 3]

        # Project gate_to_drone onto gate's forward direction
        curr_forward_projection = torch.sum(gate_to_drone * gate_forward, dim=-1)  # [32, 1]
        prev_forward_projection = torch.sum(prev_next_gate_drone_rpos * gate_forward, dim=-1)  # [32, 1]

        self.distance_to_gate_plane = curr_forward_projection

        # Check if drone has crossed the gate plane
        self.crossed_gate_plane = (
            # Drone was behind gate in previous step (negative projection)
            (prev_next_gate_drone_rpos is not None)
            & (prev_forward_projection < 0)
            # Drone is in front of gate now (positive projection)
            & (curr_forward_projection > 0)
        )

        # Check if drone passed through gate opening
        # Project gate_to_drone onto gate's up and right vectors
        gate_up = gate_rot[..., 2]  # [32, 1, 3]
        gate_right = gate_rot[..., 1]  # [32, 1, 3]

        vertical_offset = torch.abs(torch.sum(gate_to_drone * gate_up, dim=-1))
        horizontal_offset = torch.abs(torch.sum(gate_to_drone * gate_right, dim=-1))

        self.offset_to_gate_center = torch.stack((horizontal_offset, vertical_offset), dim=-1)

        # Define gate dimensions (adjust these values based on your gate size)
        gate_height = 1.0 * self.gate_scale
        gate_width = 1.0 * self.gate_scale

        self.within_gate_tunnel = (
            (vertical_offset < gate_height/2)
            & (horizontal_offset < gate_width/2)
        )

        self.valid_passing = (
            self.crossed_gate_plane
            & (vertical_offset < gate_height/2)
            & (horizontal_offset < gate_width/2)
        )

        obs = [
            self.drone_state[..., 3:],
            self.target_drone_rpos,
            self.gate_drone_rpos,
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        self.pos_error = torch.norm(self.target_drone_rpos, dim=-1)
        self.stats["pos_error"].mul_(self.alpha).add_((1-self.alpha) * self.pos_error)
        self.stats["drone_uprightness"].mul_(self.alpha).add_((1-self.alpha) * self.drone_up[..., 2])

        self.prev_drone_pos = drone_pos

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
        # crossed_plane = self.drone.pos[..., 1] < 0.
        crossed_plane = self.crossed_gate_plane
        crossing_plane = (crossed_plane & (~self.crossed_plane))
        self.crossed_plane |= crossed_plane

        # distance_to_gate_plane = 0. - self.drone.pos[..., 1]
        distance_to_gate_plane = self.distance_to_gate_plane

        # TODO: CHANGE TO NEXT GATE POSITION
        # distance_to_gate_center = torch.abs(self.drone.pos[..., [0, 2]] - self.gates_pos[0][..., [0, 2]])
        distance_to_gate_center = self.offset_to_gate_center

        through_gate = (distance_to_gate_center < 0.5).all(-1)   # TODO: CHANGE OFFSET

        reward_gate = torch.where(
            distance_to_gate_plane > 0.,
            (0.4 - distance_to_gate_center).sum(-1) * torch.exp(-distance_to_gate_plane),
            1.
        )

        distance_to_target = torch.norm(self.target_drone_rpos, dim=-1)

        # progress reward
        target_drone_rpos = self.target_pos - self.drone_state[..., :3]
        distance_to_target = torch.norm(target_drone_rpos, dim=-1)
        prev_distance_to_target = torch.norm(self.prev_target_drone_rpos, dim=-1)
        progress_reward = prev_distance_to_target - distance_to_target


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
            collision_reward = collision.float()

        # if collision.any():
        #     print(collision.nonzero(as_tuple=True)[0])
        #     import pdb; pdb.set_trace()

            # self.stats["collision"].add_(collision_reward)
        assert reward_pos.shape == reward_up.shape == reward_spin.shape

        reward = (
            reward_pos
            + 0.5 * reward_gate
            + (reward_pos + 0.3) * (reward_up + reward_spin)
            + reward_effort
            + 100 * progress_reward
            - 0.5 * collision_reward
        ) # * (1 - collision_reward)

        misbehave = (
            (self.drone.pos[..., 2] < 0.2)
            | (self.drone.pos[..., 2] > 2.5)
            # | (self.drone.pos[..., 1].abs() > 10.)   # TODO: CHANGE TERMINATION RANGE
            | (distance_to_target > 10.)    # TODO: CHANGE
        )
        hasnan = torch.isnan(self.drone_state).any(-1)
        invalid = (crossing_plane & ~through_gate)

        terminated = misbehave | hasnan | invalid
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        if self.reset_on_collision:
            terminated |= collision

        reached_target = distance_to_target < 0.2  # TODO CHANGE TO CROSSING GATE
        # if reached_target.any():
        #     print(reached_target.nonzero(as_tuple=True)[0])
        #     import pdb; pdb.set_trace()
        final_gate_index = len(self.gates_config) - 1
        not_final_gate = self.next_gate_idx < final_gate_index
        self.next_gate_idx[reached_target & not_final_gate] += 1
        # print(self.next_gate_idx)

        final_target_pos = torch.tensor(self.gates_config[-1]["pos"], device=self.device)

        distance_to_final_target = torch.norm(final_target_pos - self.drone_state[..., :3], dim=-1)
        reached_final_target = distance_to_final_target < 0.2   # TODO CHANGE TO CROSSING GATE
        # self.next_gate_idx[reached_final_target] = 0

        self.stats["success"].bitwise_or_(reached_final_target)
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
