import torch
import os
import hydra
from omegaconf import OmegaConf
from omni_drones import init_simulation_app

@hydra.main(version_base=None, config_path=".", config_name="eval2")
def evaluate_checkpoint(cfg):
    # Access the checkpoint path from the configuration
    checkpoint_path = cfg.checkpoint_path

    # Initialize simulation app first
    simulation_app = init_simulation_app(cfg)
    
    # Move these imports here, after simulation app initialization
    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.learning import ALGOS
    from omni_drones.utils.torchrl.transforms import ravel_composite, FromMultiDiscreteAction, FromDiscreteAction
    from omni_drones.utils.torchrl import RenderCallback
    from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
    from torchrl.envs.utils import set_exploration_type, ExplorationType

    # Initialize simulation app and environment
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    # Set up transforms
    transforms = [InitTracker()]
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    action_transform = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)

    env = TransformedEnv(base_env, Compose(*transforms)).eval()

    # Load policy
    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device
    )

    # Load the checkpoint
    policy.load_state_dict(torch.load(checkpoint_path))

    # Define the evaluation function
    @torch.no_grad()
    def evaluate(seed: int=0, exploration_type: ExplorationType=ExplorationType.MODE):
        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        # log video
        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"),
            fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
            format="mp4"
        )

        return info

    # Run evaluation
    info = evaluate()
    print(info)

    simulation_app.close()

if __name__ == "__main__":
    evaluate_checkpoint()