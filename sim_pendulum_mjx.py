import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.registration import register

# Import your custom environment
from inverted_double_pendulum_mjx import InvertedDoublePendulumEnv

# Register your custom environment (if not already registered)
register(
    id="MJXCustomDoublePendulum-v0",
    entry_point="inverted_double_pendulum_mjx:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
)

n_envs = 32768

# Define different numbers of steps for the experiment
step_counts = [100, 200, 500, 1000, 3000, 5000, 7000, 10000]
times_taken = []

for steps in step_counts:
    # Create a new instance of the environment for each trial
    env = gym.make("MJXCustomDoublePendulum-v0", n_envs=n_envs)
    env.reset()

    step = 0
    start = time.time()
    # Run the simulation until the desired number of steps is reached
    while step < steps:
        # Sample n_envs actions
        action = env.action_space.sample()
        action_batch = np.tile(action, (n_envs, 1))

        obs, reward, done, truncated, info = env.step(action_batch)
        step += 1
        # Optionally, handle episode termination if needed:
    elapsed = time.time() - start
    times_taken.append(elapsed)
    print(f"Steps: {steps}, Time taken: {elapsed:.4f} sec")
    env.close()

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(step_counts, times_taken, marker="o")
plt.xlabel("Number of Steps")
plt.ylabel("Time Taken (sec)")
plt.title("Benchmarking MJX Simulation")
# Compute best fit line and plot it
m, b = np.polyfit(step_counts, times_taken, 1)
plt.plot(step_counts, [m * x + b for x in step_counts], "--", color="gray")
plt.legend(["Time Taken", f"Best Fit Line: {m:.4f}x + {b:.4f}"])

plt.grid(True)
plt.savefig("benchmark_mjx.png")
