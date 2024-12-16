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
# draw_gate_arrow, TrackLoader, ellipse, generate_smooth_trajectory
import json


class Racing(IsaacEnv):
    r"""
    An advanced control task where the agent should fly the UAV through multiple gates to complete the racing.
    = This task is a reproduction of the racing task in the paper "Reaching the limit in autonomous racing: Optimal control vs RL".
    - Implemented by Tony Wang & Jerry Wang as final project for the course "CIS519: Applied Machine Learning" in University of Pennsylvania.

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
        self.trajectory_type = cfg.task.trajectory_type
        
       
        map_path = f"/home/tonyw/Projects/RL_drone/OD_main/cfg/task/RacingTrack/{self.trajectory_type}.yaml"
        # map_path = "./split_s.yaml"
        with open(map_path, 'r') as f:
            gates_config_yaml = yaml.safe_load(f)
        self.gates_config = load_gates_from_yaml(gates_config_yaml)

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

        self.next_gate_idx = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.long)
        # self.prev_next_gate_pos = None
        # self.prev_next_gate_drone_rpos = None
        self.prev_drone_pos = self.drone.get_state()[..., :3]

        self.init_pos_dist = D.Uniform(
            torch.tensor([-1.0, 12.0, 1.5], device=self.device),
            torch.tensor([0, 12.0, 2.5], device=self.device)
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
            "/World/defaultGroundPlane", # TODO can not render
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            color=(0.4, 0.26, 0.13),  # Brown wood-like color
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

        self.drone.spawn(translations=[(8.0, 2.0, 3.0)])

        target = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=(5.0, 0., 2.),
            radius=0.1,
            color=torch.tensor([1., 0., 0.]), # red
        )
        kit_utils.set_collision_properties(target.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(target.prim_path, disable_gravity=True)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        # observation_dim = drone_state_dim + 6
        observation_dim = drone_state_dim
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
        # print("Stats return shape:", self.stats["return"].shape)

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

        drone_state = self.drone.get_state()
        self.prev_drone_pos = drone_state[..., :3] 

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
                draw_gate_arrow(self.draw,gate_cfg)


            if self.trajectory_type == "ellipse": 
                t = torch.linspace(0, 2*torch.pi, 100, device=self.device)
                self.trajectory = ellipse(t)
            else:
                self.trajectory = generate_smooth_trajectory(self.gates_config)
            self.trajectory = torch.tensor(self.trajectory, device=self.device)  

            if self.trajectory_type == "UZH_track": 
                traj_points = self.trajectory.unsqueeze(0)
            else:
                traj_points = self.trajectory
            point_list_0 = traj_points[:-1].tolist()
            point_list_1 = traj_points[1:].tolist()
            colors = [(0.0, 1.0, 1.0, 0.5) for _ in range(len(point_list_0))] # Cyan translucency
            sizes = [3 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

             # 画管道边界（上下各画一条线来表示管道）
            tube_radius = 2.2
            # 上边界
            upper_points = traj_points + tube_radius * torch.tensor([0., 0., 1.], device=self.device)
            point_list_0 = upper_points[:-1].tolist()
            point_list_1 = upper_points[1:].tolist()
            colors = [(1.0, 0.0, 0.0, 0.3) for _ in range(len(point_list_0))]  # 红色半透明
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
            
            # 下边界
            lower_points = traj_points - tube_radius * torch.tensor([0., 0., 1.], device=self.device)
            point_list_0 = lower_points[:-1].tolist()
            point_list_1 = lower_points[1:].tolist()
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)
        # current_time = float(self.progress_buf * self.dt)
        # rotation = euler_to_quaternion(
        #     torch.tensor([0., current_time, 0.], device=self.device)
        # ).expand(self.num_envs, 1, 4)
        # self.target.set_world_poses(
        #     positions=self.target_pos + self.envs_positions.unsqueeze(1),
        #     orientations=rotation
        # )


    # Observation space
    def _compute_state_and_obs(self):
        self.prev_next_gate_pos = self.next_gate_pos if hasattr(self, 'next_gate_pos') else None

        self.drone_state = self.drone.get_state()
        self.drone_up = self.drone_state[..., 16:19]
        # print("self.drone_state shape:", self.drone_state.shape)
        # print("self.drone_up shape:", self.drone_up.shape)

        # 1. Progress reward
        # 1.1 get gate position
        self.gates_pos = []
        self.gates_ori = []
        for gate in self.gates:
            gate_pos = self.get_env_poses(gate.get_world_poses())[0]  # shape: [32, 1, 3]
            gate_ori = self.get_env_poses(gate.get_world_poses())[1]  # shape: [32, 1, 4]
            self.gates_pos.append(gate_pos)
            self.gates_ori.append(gate_ori)
        
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

        # 1.3 get next gate position and orientation
        self.next_gate_pos = gates_pos_stack[batch_indices, self.next_gate_idx.squeeze(-1)]  # shape: [32, 1, 3]
        self.next_gate_ori = torch.stack(self.gates_ori, dim=1)[batch_indices, self.next_gate_idx.squeeze(-1)]  # shape: [32, 1, 4]

        # Check if drone has passed through the next gate
        drone_pos = self.drone_state[..., :3]  # shape: [32, 1, 3]
        
        # Convert gate orientation to rotation matrix
        gate_rot = quat_to_matrix(self.next_gate_ori)  # [32, 1, 3, 3]
        
        # Get gate's forward direction (assuming gate's forward is along local X axis)
        gate_forward = gate_rot[..., 0]  # [32, 1, 3]

        # distance between drone position at the last timestep and next gate position
        prev_next_gate_drone_rpos = self.prev_drone_pos - self.next_gate_pos

        # Vector from gate to drone
        gate_to_drone = drone_pos - self.next_gate_pos  # [32, 1, 3]
        
        # Project gate_to_drone onto gate's forward direction
        curr_forward_projection = torch.sum(gate_to_drone * gate_forward, dim=-1)  # [32, 1]
        prev_forward_projection = torch.sum(prev_next_gate_drone_rpos * gate_forward, dim=-1)  # [32, 1]
        
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
        
        # Define gate dimensions (adjust these values based on your gate size)
        gate_height = 1.0 * self.gate_scale
        gate_width = 1.0 * self.gate_scale

        self.passed_within_gate = (
            (vertical_offset < gate_height/2) 
            & (horizontal_offset < gate_width/2)
        )
        
        self.valid_passing = (
            self.crossed_gate_plane 
            & (vertical_offset < gate_height/2) 
            & (horizontal_offset < gate_width/2)
        )
        
        # Update next gate index when drone passes through current gate
        self.next_gate_idx = torch.where(
            self.valid_passing,
            (self.next_gate_idx + 1) % len(self.gates_config),
            self.next_gate_idx
        )

        # 2. relative position
        # 2.1 target position
        self.target_drone_rpos = self.target_pos - drone_pos  # shape: [32, 1, 3]
        # self.gate_drone_rpos = self.nearest_gate_pos - drone_pos  # shape: [32, 1, 3]
        self.next_gate_drone_rpos = self.next_gate_pos - drone_pos  # shape: [32, 1, 3]

        # 3. line following
        


        obs = [
            self.drone_state[..., 3:],  # shape: [32, 1, 20]
            # self.target_drone_rpos,     # shape: [32, 1, 3]
            self.next_gate_drone_rpos,       # shape: [32, 1, 3]
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        
        obs = torch.cat(obs, dim=-1)

        self.pos_error = torch.norm(self.next_gate_drone_rpos, dim=-1)
        self.stats["pos_error"].mul_(self.alpha).add_((1-self.alpha) * self.pos_error)
        self.stats["drone_uprightness"].mul_(self.alpha).add_((1-self.alpha) * self.drone_up[..., 2])

        self.prev_drone_pos = self.drone_state[..., :3]

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    # Reward function
    def _compute_reward_and_done(self):
        lambda_progress = 5.0
        lambda_cmd1 = -2e4
        lambda_cmd2 = -1e4
        lambda_traj = 100 # 轨迹跟踪奖励权重

        # progress reward
        rpos_prev = self.prev_drone_pos - self.next_gate_pos
        rpos_curr = self.drone_state[..., :3] - self.next_gate_pos
        d_prev = torch.norm(rpos_prev, dim=-1) # shape: [num_envs, 1]
        d_curr = torch.norm(rpos_curr, dim=-1) # shape: [num_envs, 1]
        reward_progress = (d_prev - d_curr).sum(-1, keepdim=True)  # shape: [num_envs, 1]
        # import pdb; pdb.set_trace()

        # # uprightness
        reward_up = 0.05 * torch.square((self.drone_up[..., 2] + 1) / 2)
        # effort
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 0.5 / (1.0 + torch.square(spin))

        collision_reward = torch.zeros((self.num_envs, 1), device=self.device)

        if self.reset_on_collision:
            collision = (
                self.gate_frames[0]
                .get_net_contact_forces()
                .any(-1)
                .any(-1, keepdim=True)
            )
            collision_reward = torch.where(
                collision,  # shape: [num_envs, 1]
                torch.full_like(collision, -5.0, dtype=torch.float),  # if True
                torch.zeros_like(collision, dtype=torch.float)  # if False
        )

        # Line tracking reward
        if self.trajectory_type == "ellipse": 
            t = torch.linspace(0, 2*torch.pi, 100, device=self.device)
            self.trajectory = ellipse(t)
        else:
            self.trajectory = generate_smooth_trajectory(self.gates_config)
        self.trajectory = torch.tensor(self.trajectory, device=self.device)  

        drone_pos = self.drone_state[..., :3] 
        drone_pos = drone_pos.squeeze(1)       # [num_envs, 3] 
        assert self.trajectory.device == drone_pos.device, f"Device mismatch: trajectory on {self.trajectory.device}, drone_pos on {drone_pos.device}"

        distances = torch.norm(self.trajectory.unsqueeze(0) - drone_pos.unsqueeze(1), dim=-1).squeeze() # [num_envs, 100]
        min_distances = torch.min(distances, dim=-1)[0] # [num_envs]
        reward_traj = lambda_traj * torch.exp(-2.0 * min_distances).unsqueeze(-1) # [num_envs, 1]
        # print("t shape:", t.shape)
        # print("traj_points shape:", traj_points.shape)
        # print("drone_pos shape:", drone_pos.shape)
        # print("distances shape:", distances.shape)
        # print("min_distances shape:", min_distances.shape)
        # print("reward_traj shape:", reward_traj.shape)

        # import pdb; pdb.set_trace()


        gate_reward = torch.zeros_like(reward_progress)  # [num_envs, 1]
        gate_passing_bonus = 5.0
        if self.valid_passing.any():
            # print(f"self.valid_passing shape: {self.valid_passing.shape}")
            # print(f"self.valid_passing value: {self.valid_passing}")
            gate_reward[self.valid_passing] = gate_passing_bonus

        # speed reward
        current_vel = self.drone.vel[..., :3]  # [num_envs, 1, 3]
        desired_speed = 30.0  # TODO how to get really fast?
        speed = torch.norm(current_vel, dim=-1)  # [num_envs, 1]
        speed_reward = torch.exp(-0.5 * (speed - desired_speed)**2)  # [num_envs, 1]
        
        # print(f"speed_reward shape: {speed_reward.shape}")
        pass_reward = 0.5 * self.next_gate_idx / len(self.gates_config)  # [num_envs, 1]
        
        reward = (
            lambda_progress * reward_progress
            + reward_effort
            + reward_up
            # + collision_reward
            + reward_spin
            + reward_traj
            # + gate_reward
            # + speed_reward
            # + pass_reward
        )   
        # import pdb; pdb.set_trace()
        
        # TODO: add bounding box for race track
        tube_radius = 2.2 
        out_of_bounds = (min_distances > tube_radius).unsqueeze(-1)  # [num_envs, 1]
        boundary_penalty = torch.zeros_like(reward_progress)  # [num_envs, 1]
        near_boundary = (min_distances > (tube_radius * 0.8)).unsqueeze(-1)
        boundary_penalty[near_boundary] = -2.0  
        reward = reward + boundary_penalty
        
        misbehave = (
            (self.drone.pos[..., 2] < 0.2)
            | (self.drone.pos[..., 2] > 4)
            | out_of_bounds
        )

  

        hasnan = torch.isnan(self.drone_state).any(-1)
        invalid = (self.crossed_gate_plane & ~self.passed_within_gate)
        terminated = misbehave | hasnan | invalid
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        if self.reset_on_collision:
            terminated |= collision

        self.stats["success"].bitwise_or_(self.valid_passing)
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        # import pdb; pdb.set_trace()
        return TensorDict(
            {
                "agents": {
                    "reward": reward,
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
