import pathlib
import time
from typing import NamedTuple

import optax
from flax.serialization import to_bytes
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
from wrappers import (
    ClipAction,
    InvertedDoublePendulumGymnaxWrapper,
    NormalizeVecObservation,
    NormalizeVecReward,
    VecEnv,
)

import jax
import jax.experimental
import jax.numpy as jnp
import wandb
from models import ActorCritic, save_model

LOG_DIR = pathlib.Path("logs")

config = {
    "LR": 3e-4,
    "NUM_ENVS": 4096,
    "NUM_STEPS": 10,
    "TOTAL_TIMESTEPS": 8e8,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 32,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.000,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ENV_NAME": "hopper",
    "ANNEAL_LR": True,
    "NORMALIZE_ENV": True,
    "LOGGING": True,
}

steps_elapsed = 0


def callback(info, loss_info):
    global steps_elapsed, config

    done = info["returned_episode"]
    returns = info["returned_episode_returns"]
    lengths = info["returned_episode_lengths"]

    # Unpack loss info
    value_loss = loss_info["value_loss"]
    actor_loss = loss_info["actor_loss"]
    entropy = loss_info["entropy"]
    pred_loss = loss_info["pred_loss"]

    # Only consider environments where episode ended during this step
    mask = done.astype(bool)

    if jnp.any(mask):  # Check if there are any "done" environments
        # Flatten across time and envs
        episode_returns = returns[mask]

        total_return = jnp.sum(episode_returns)
        num_episodes = jnp.sum(mask)

        avg_return = total_return / num_episodes

        episode_lengths = lengths[mask]
        avg_length = jnp.sum(episode_lengths) / num_episodes

        print(f"Step {steps_elapsed}:")
        print(f"  Average return: {avg_return}")
        print(f"  Average length: {avg_length}")
        print(f"  Value loss: {value_loss}")
        print(f"  Actor loss: {actor_loss}")
        print(f"  Entropy: {entropy}")

        # Log average episode return and length to wandb
        wandb.log(
            {
                "rollout/ep_rew_mean": avg_return,
                "rollout/ep_len_mean": avg_length,
                "global_step": steps_elapsed * config["NUM_ENVS"] * config["NUM_STEPS"],
                "train/value_loss": value_loss,
                "train/actor_loss": actor_loss,
                "train/entropy": entropy,
                "train/pred_loss": pred_loss,
            }
        )
    else:
        print(f"No episodes completed during this step: {steps_elapsed}")

    steps_elapsed += 1


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env, env_params = InvertedDoublePendulumGymnaxWrapper(), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    # def linear_schedule(count):
    #     frac = (
    #         1.0
    #         - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
    #         / config["NUM_UPDATES"]
    #     )
    #     return config["LR"] * frac

    def linear_schedule(count):
        total_steps = (
            config["NUM_UPDATES"] * config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
        )
        step_frac = count / total_steps

        x = (step_frac - 0.2) / 0.8
        x = jnp.clip(x, 0.0, 1.0)

        lr_start = 3e-4
        # lr_end = 8e-5
        lr_end = 8e-5

        lr = jnp.exp(x * jnp.log(lr_end) + (1 - x) * jnp.log(lr_start))
        return lr

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )

                info_filtered = {"termination": info["termination"]}
                info_filtered["returned_episode"] = info["returned_episode"]
                info_filtered["returned_episode_returns"] = info[
                    "returned_episode_returns"
                ]
                info_filtered["returned_episode_lengths"] = info[
                    "returned_episode_lengths"
                ]

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info_filtered
                )

                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                # Use both terminated and truncated
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    terminated = transition.info["termination"]
                    # done = transition.done

                    value = transition.value
                    reward = transition.reward

                    # This is a subtle distinction:
                    # If not done, bootstrap (next_value)
                    # If episode is truncated, bootstrap (next_value)
                    # If episode is terminated, use 0 as the next value
                    not_final = 1.0 - terminated  # 0 if terminated, 1 otherwise

                    delta = reward + config["GAMMA"] * next_value * not_final - value
                    gae = (
                        delta + config["GAMMA"] * config["GAE_LAMBDA"] * not_final * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            if config.get("LOGGING", False):
                jax.experimental.io_callback(callback, None, metric, loss_info)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)

        runner_state, _ = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": None}

    return train


def main():
    global config, LOG_DIR

    # Check if log directory exists
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    if config["LOGGING"]:
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

        # Log all hyperparameters
        wandb.config.update(config)
        wandb.run.save()
        print("Wandb run started")

    rng = jax.random.PRNGKey(74837483)

    train_jit = jax.jit(make_train(config))
    start_time = time.time()

    out = train_jit(rng)

    # Wait for all JAX computations to finish
    jax.block_until_ready(out)

    print("Total training time: ", time.time() - start_time)

    train_state = out["runner_state"][0]
    params = train_state.params

    if config["NORMALIZE_ENV"]:
        env_state = out["runner_state"][1]
        obs_mean = env_state.env_state.mean[0]
        obs_var = env_state.env_state.var[0]

        # Write the normalization parameters to a file
        norm_params_path = pathlib.Path("norm_params.msgpack")
        with norm_params_path.open("wb") as f:
            f.write(to_bytes((obs_mean, obs_var)))
        print("Normalization parameters saved to: ", norm_params_path)
    else:
        obs_mean = jnp.zeros((1,))
        obs_var = jnp.ones((1,))

    if config["LOGGING"]:
        save_path = LOG_DIR / f"{run.name}_final_policy.zip"
        save_model(params, obs_mean, obs_var, save_path)

        wandb.finish()
        print("Wandb run finished")
    print("Final policy saved to: ", save_path)


if __name__ == "__main__":
    main()
