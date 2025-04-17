import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy as media

from inverted_double_pendulum_mjx import InvertedDoublePendulumEnv

# Create a single environment instance.
env = InvertedDoublePendulumEnv()

# Number of parallel environments to simulate.
num_envs = 4096

# JIT-compile the reset and step functions.
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# Create a base RNG key.
base_rng = jax.random.PRNGKey(42)

# Initialize an RNG key for each environment.
init_rngs = jax.random.split(base_rng, num_envs)

# Reset all environments in parallel using vmap.
# 'states' is a batched array of environment states.
states = jax.vmap(jit_reset)(init_rngs)

# Set up rollout parameters.
num_steps = 300
# Save all states across time (batched; shape: (num_envs, ...))
rollouts = [states]


# List of different numbers of environments to simulate.
num_envs_list = [256, 1024, 4096, 8192, 32768]
simulation_times = []

for num_envs in num_envs_list:
    # Reinitialize RNGs and states for the current num_envs.
    init_rngs = jax.random.split(base_rng, num_envs)
    states = jax.vmap(jit_reset)(init_rngs)

    # Measure simulation time.
    start = time.time()
    rng = base_rng
    for t in range(num_steps):
        # Generate actions for each environment.
        rng, action_rng = jax.random.split(rng)
        action_rngs = jax.random.split(action_rng, num_envs)
        ctrls = jax.vmap(
            lambda r: jax.random.uniform(r, (1,), minval=-1.0, maxval=1.0)
        )(action_rngs)

        # Update each environment in parallel.
        states = jax.vmap(jit_step)(states, ctrls)

    simulation_time = time.time() - start
    simulation_times.append(num_steps * num_envs / simulation_time)
    print(f"num_envs: {num_envs}, simulation time: {simulation_time:.2f} seconds")


# Plot the results.
plt.figure(figsize=(10, 6))
plt.plot(num_envs_list, simulation_times, marker="o")
plt.xlabel("Number of Environments (num_envs)")
plt.ylabel("Steps per Second")
plt.title("Steps/Second vs Number of Environments")

plt.savefig("simulation_time_vs_num_envs.png")
