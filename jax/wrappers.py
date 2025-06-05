from flax import struct
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper
from inverted_double_pendulum_mjx import InvertedDoublePendulumEnv
from mujoco_playground import wrapper

import jax
import jax.numpy as jnp


class InvertedDoublePendulumGymnaxWrapper:
    def __init__(self, mode_switch_steps=1000, switch_order=None):
        env = InvertedDoublePendulumEnv(
            mode_switch_steps=mode_switch_steps, switch_order=switch_order
        )
        env = wrapper.wrap_for_brax_training(
            env,
            vision=False,
            episode_length=2000,
            action_repeat=1,
            # randomization_fn=training_params.get("randomization_fn"),
        )
        # env = envs.get_environment(env_name=env_name, backend=backend)
        # env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        # env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        # Update info with new rng key
        state.info["rng"] = key
        next_state = self._env.step(state, action)
        return (
            next_state.obs,
            next_state,
            next_state.reward,
            next_state.done > 0.5,
            {
                # "dead_steps": next_state.info["dead_steps"],
                # "mode_switch_steps": next_state.info["mode_switch_steps"],
                "termination": next_state.done > 0.5,
                "was_reset": next_state.info["was_reset"],
            },
        )

    def observation_space(self, params):
        return self._env.observation_space
        # return spaces.Box(
        #     low=-jnp.inf,
        #     high=jnp.inf,
        #     shape=(self._env.observation_size,),
        # )

    def action_space(self, params):
        return self._env.action_space
        # return spaces.Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(self._env.action_size,),
        # )

    def render(self, state):
        return self._env.render(state)


@struct.dataclass
class TimeOffsetState:
    prev_action: jnp.ndarray
    prev_obs: jnp.ndarray
    env_state: environment.EnvState


class TimeOffset(GymnaxWrapper):
    """Time Offset shifts the observation by 1 step."""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)

        # Generate a 0 action
        prev_obs = jnp.zeros_like(obs)
        prev_action = jnp.zeros((1,))

        state = TimeOffsetState(
            prev_action=prev_action,
            prev_obs=obs,
            env_state=state,
        )

        # Combine the two observations
        # obs = jnp.concatenate((prev_obs, prev_action), axis=-1)
        # obs = jnp.concatenate((obs, prev_action, obs), axis=-1)

        return obs, state

    def step(self, key, state, action, params=None):
        # Update the info with the new rng key
        new_obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        # Update the observation
        # If the environment was reset, we want to use the new observation
        # Otherwise, we want to use the previous observation
        # obs = jax.lax.cond(
        #     info["was_reset"] > 0.5,
        #     lambda _: jnp.concatenate(
        #         (jnp.zeros_like(new_obs), jnp.zeros((1,))), axis=-1
        #     ),
        #     lambda _: jnp.concatenate((state.prev_obs, state.prev_action), axis=-1),
        #     operand=None,
        # )
        # obs = jnp.concatenate((new_obs, state.prev_action, new_obs), axis=-1)

        state = TimeOffsetState(
            prev_action=action,
            prev_obs=new_obs,
            env_state=env_state,
        )

        # Value function should get the most recent observation
        # info["value_observation"] = jnp.concatenate((prev_obs, new_obs), axis=-1)
        info["value_observation"] = new_obs

        return new_obs, state, reward, done, info


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        # info["value_observation"] contains a observation for the value function
        # Normalize it as well
        # Last dimension of state.mean and state.var is 1 element longer than value_obs
        # because of the action dimension
        # Remove the last element of state.mean and state.var
        # val_mean = state.mean[..., :-1]
        # val_var = state.var[..., :-1]
        # info["value_observation"] = (info["value_observation"] - val_mean) / jnp.sqrt(
        #     val_var + 1e-8
        # )

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info
