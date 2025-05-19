import pathlib
from functools import partial

import imageio
from flax.serialization import from_bytes
from wrappers import (
    ClipAction,
    InvertedDoublePendulumGymnaxWrapper,
    NormalizeVecObservation,
    NormalizeVecReward,
    VecEnv,
)

import jax
import jax.numpy as jnp
from models import ActorCritic

# -------- Reconstruct environment --------

mode_order = [0, 1, 2, 3]
mode_switch_steps = 800
episode_length = mode_switch_steps * len(mode_order)

env = InvertedDoublePendulumGymnaxWrapper(
    mode_switch_steps=mode_switch_steps, switch_order=mode_order
)
env = ClipAction(env)

# -------- Load trained policy --------
model = ActorCritic(action_dim=env.action_space(None).shape[0], activation="tanh")
params_path = pathlib.Path("final_policy.msgpack")
params = from_bytes(
    model.init(jax.random.PRNGKey(0), jnp.zeros(env.observation_space(None).shape)),
    params_path.read_bytes(),
)


obs_dim = env.observation_space(None).shape[0]
norm_params_path = pathlib.Path("norm_params.msgpack")
with norm_params_path.open("rb") as f:
    data = f.read()
empty = (jnp.zeros(obs_dim), jnp.ones(obs_dim))
obs_mean, obs_var = from_bytes(empty, data)

# --- Rollout ---
rng = jax.random.PRNGKey(0)
n_episodes = 1
episode_length = 800  # or env_cfg.episode_length if you have it

frames = []
rewards = []


@partial(jax.jit, static_argnums=2)
def run_rollout(rng, params, episode_length):
    def step_fn(carry, _):
        rng, state = carry
        rng, key = jax.random.split(rng)

        # Normalize observation
        obs = state.obs
        obs = (obs - obs_mean) / jnp.sqrt(obs_var + 1e-8)

        pi, _ = model.apply(params, obs)
        action = pi.mean()
        obs, next_state, reward, done, info = env.step(key, state, action)
        carry = (rng, next_state)
        return carry, (next_state, reward, done)

    key, reset_key = jax.random.split(rng)
    obs, state = env.reset(reset_key)
    initial_state = state
    carry = (key, state)

    carry, (states, rewards, dones) = jax.lax.scan(
        step_fn, carry, None, length=episode_length
    )

    # Prepend initial state so total = episode_length + 1
    # full_states = jax.tree_util.tree_map(
    #     lambda x: jnp.concatenate([x[:1].at[0].set(initial_state), x], axis=0),
    #     states,
    # )

    return states, rewards, dones


# Convert from batched PyTree → list of per-step State objects
def unbatch_states(batched_states):
    # Unzip time dimension into list of field-wise dicts
    return [
        jax.tree_util.tree_map(lambda x: x[i], batched_states)
        for i in range(batched_states.obs.shape[0])
    ]


# --- Execute rollout ---
rng = jax.random.PRNGKey(200)
episode_length = 2400
states, rewards, dones = run_rollout(rng, params, episode_length)


state_list = unbatch_states(states)

# --- Find termination point (first done=True)

# --- Render rollout ---
render_every = 1
frames = [env.render(s) for s in state_list[::render_every]]

# --- Save video ---
video_path = pathlib.Path("rollout.mp4")
imageio.mimsave(video_path, frames, fps=40)
print(f"Saved rollout to {video_path}")
print(f"Total return: {sum(rewards):.2f}, Episode length: {len(rewards)}")
