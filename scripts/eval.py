import logging
import os
import hydra
import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from omni_drones import init_simulation_app
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback
from omni_drones.learning import ALGOS

from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


@hydra.main(version_base=None, config_path=".", config_name="eval")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    
    if cfg.get("use_wandb", False):
        run = init_wandb(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # Apply the same transforms as during training
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms))
    env.set_seed(cfg.seed)

    # Add after env creation
    print("Observation spec:", env.observation_spec)
    print("Reward spec:", env.reward_spec)
    print("Action spec:", env.action_spec)

    # Initialize policy
    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    # Load checkpoint
    if cfg.checkpoint_path:
        checkpoint = torch.load(cfg.checkpoint_path, map_location=base_env.device)
        policy.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {cfg.checkpoint_path}")

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        num_episodes: int=10,
        exploration_type: ExplorationType=ExplorationType.MODE,
        save_video: bool=True
    ):
        base_env.enable_render(True)
        env.eval()
        env.set_seed(seed)

        episode_returns = []
        episode_lengths = []
        
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            render_callback = RenderCallback(interval=2)
            
            with set_exploration_type(exploration_type):
                trajs = env.rollout(
                    max_steps=base_env.max_episode_length,
                    policy=policy,
                    callback=render_callback,
                    auto_reset=True,
                    break_when_any_done=True,
                    return_contiguous=False,
                )

            # Get episode statistics
            done = trajs.get(("next", "done"))
            try:
                rewards = trajs.get(("next", "reward"))  # Some environments store reward in next
            except KeyError:
                try:
                    rewards = trajs.get(("agents", "reward"))  # Others might store it in agents
                except KeyError:
                    print("Available keys in trajectory:", trajs.keys(True, True))
                    raise KeyError("Could not find reward in trajectory data")
            
            episode_return = rewards.sum().item()
            episode_length = len(rewards)
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

            if save_video and episode == 0:  # Save video for first episode
                video_array = render_callback.get_video_array(axes="t c h w")
                if cfg.get("use_wandb", False):
                    wandb.log({
                        "eval_video": wandb.Video(
                            video_array,
                            fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                            format="mp4"
                        )
                    })

        stats = {
            "eval/mean_return": sum(episode_returns) / len(episode_returns),
            "eval/mean_episode_length": sum(episode_lengths) / len(episode_lengths),
            "eval/std_return": torch.tensor(episode_returns).std().item(),
        }

        if cfg.get("use_wandb", False):
            wandb.log(stats)
        
        print("\nEvaluation Results:")
        print(OmegaConf.to_yaml(stats))
        
        return stats

    # Run evaluation
    evaluate(
        seed=cfg.seed,
        num_episodes=cfg.num_eval_episodes,
        save_video=cfg.save_video
    )

    if cfg.get("use_wandb", False):
        wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
