# This script is just to run through the motions of training an agent for the air hockey challenge

import math

import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

import wandb

register(
    id="CustomDoublePendulum-v0",
    entry_point="inverted_double_pendulum:InvertedDoublePendulumEnv",
    max_episode_steps=2000,
)

# env = gym.make("CustomDoublePendulum-v0")


def main():
    wandb.login()

    run = wandb.init(
        # Set the project where this run will be logged
        project="double-pendulum",
        # Track hyperparameters and run metadata
        config={},
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optiona
    )

    n_envs = 20

    # Parallel environments
    vec_env = make_vec_env(
        "CustomDoublePendulum-v0", n_envs=n_envs, vec_env_cls=SubprocVecEnv
    )

    # Save a checkpoint every 1000 steps
    experiment_name = run.name
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path="./logs/",
        name_prefix=experiment_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256, 256, 256, 256], vf=[256, 256, 256, 256, 256])],
    )

    timesteps = 600_000_000

    def lr_decay(x):
        # Decay to 70% of training steps
        # X is the fraction of training steps remaining

        # Scale x such that 0.3 to 1 is 0 to 1
        x = (x - 0.3) / 0.7

        x = max(0, x)

        return math.exp(x * math.log(3e-4) + (1 - x) * math.log(8e-5))

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device="cuda:0",
        ent_coef=0.0006,
        tensorboard_log=f"runs/{run.id}",
        batch_size=int(n_envs * 2000 / 2),
        n_epochs=20,
        max_grad_norm=0.25,
        policy_kwargs=policy_kwargs,
        # decay learning rate to 1e-6 by the end of training
        learning_rate=lr_decay,
    )
    model.learn(
        total_timesteps=timesteps,
        callback=[
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
            checkpoint_callback,
        ],
    )

    run.finish()

    # Save the model
    model.save("ppo_double_pendulum")


if __name__ == "__main__":
    main()
