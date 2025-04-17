import jax
import jax.numpy as jnp
import mediapy as media

from inverted_double_pendulum_mjx import InvertedDoublePendulumEnv

env = InvertedDoublePendulumEnv()

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

rng = jax.random.PRNGKey(42)


state = jit_reset(jax.random.PRNGKey(0))
rollout = [state]
for i in range(300):
    act_rng, rng = jax.random.split(rng)
    # Generate a random action, between -1 and 1
    ctrl = jax.random.uniform(act_rng, (1,), minval=-1.0, maxval=1.0)

    state = jit_step(state, ctrl)
    rollout.append(state)

    print(i)

render_every = 1
frames = env.render(rollout[::render_every])
rewards = [s.reward for s in rollout]

# Save the video to a file and play it back
output_path = "/home/swapnil/efr/double-pendulum-playground/output.mp4"
media.write_video(output_path, frames, fps=1 / env.dt)
print(f"Video saved to {output_path}")
media.show_video(frames, fps=1 / env.dt)
