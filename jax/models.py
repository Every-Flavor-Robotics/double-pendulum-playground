import pathlib
import tempfile
import zipfile
from typing import Sequence

import distrax
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax.serialization import from_bytes, to_bytes

import jax.numpy as jnp


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), None


class ActorCriticAssymetric(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs_dim = x.shape[-1] // 2

        prev_obs = x[..., :obs_dim]
        action = x[..., obs_dim : obs_dim + 1]
        cur_obs = x[..., obs_dim + 1 :]

        x_pred = jnp.concatenate([prev_obs, action], axis=-1)

        state_pred = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x_pred)
        state_pred = activation(state_pred)
        state_pred = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(state_pred)
        state_pred = activation(state_pred)
        state_pred = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(state_pred)
        state_pred = activation(state_pred)
        state_pred = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(state_pred)
        state_pred = activation(state_pred)
        state_pred = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(state_pred)
        state_pred = activation(state_pred)
        state_pred = nn.Dense(
            obs_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(state_pred)

        # Construct the prediction of the current observation
        cur_obs_pred = state_pred + prev_obs

        # Construct the input for the policy
        # x_policy = lax.stop_gradient(cur_obs_pred)
        x_policy = jnp.concatenate([prev_obs, action], axis=-1)

        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x_policy)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # x_critic is just the current observation
        x_critic = cur_obs

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x_critic)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), state_pred


def save_model(params, obs_mean, obs_var, save_path):
    """Saves the model parameters and normalization parameters to temp files, and zips them to save path.

    Args:
        params : Model parameters.
        obs_mean : Observation mean.
        obs_var : Observation variance.
        save_path : Path to save the model.
    """

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the model parameters
        model_path = pathlib.Path(temp_dir) / "model.msgpack"
        with model_path.open("wb") as f:
            f.write(to_bytes(params))

        # Save the normalization parameters
        norm_params_path = pathlib.Path(temp_dir) / "norm_params.msgpack"
        with norm_params_path.open("wb") as f:
            f.write(to_bytes((obs_mean, obs_var)))

        # Zip the files
        zip_path = pathlib.Path(save_path)
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(model_path, arcname="model.msgpack")
            zipf.write(norm_params_path, arcname="norm_params.msgpack")

    print(f"Model and normalization parameters saved to: {zip_path}")


def load_model(zip_path, model, key, env):
    """Loads the model parameters and normalization parameters from a zip file.

    Args:
        zip_path : Path to the zip file containing the model and normalization parameters.

    Returns:
        params : Model parameters.
        obs_mean : Observation mean.
        obs_var : Observation variance.
    """

    obs_dim = env.observation_space(None).shape[0]

    # Put model
    with zipfile.ZipFile(zip_path, "r") as zipf:
        with zipf.open("model.msgpack") as f:
            params_bytes = f.read()
        with zipf.open("norm_params.msgpack") as f:
            obs_data = f.read()

    params = from_bytes(
        model.init(key, jnp.zeros(env.observation_space(None).shape)),
        params_bytes,
    )

    empty = (jnp.zeros(obs_dim), jnp.ones(obs_dim))
    obs_mean, obs_var = from_bytes(empty, obs_data)

    return params, obs_mean, obs_var
