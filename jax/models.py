import pathlib
import tempfile
import zipfile
from typing import Sequence

import distrax
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from flax.serialization import from_bytes, to_bytes

import jax
import jax.numpy as jnp


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, x_val):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Remove action from the input
        # Split into x and x_val
        # input_dim = x.shape[-1]

        # x_dim = (input_dim - 1) // 2

        # print(f"input_dim: {input_dim}, x_dim: {x_dim}")

        # x_policy = x[..., :x_dim]
        # x_val = x[..., x_dim:]

        # print(f"x shape: {x_policy.shape}, x_val shape: {x_val.shape}")

        x_policy = x

        state_prediction_head = nn.Dense(
            512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x_policy)
        # state_prediction_head = nn.relu(state_prediction_head)

        # state_prediction_head = nn.Dense(
        #     512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(state_prediction_head)
        # state_prediction_head = nn.relu(state_prediction_head)

        # state_prediction_head = nn.Dense(
        #     512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(state_prediction_head)
        # state_prediction_head = nn.relu(state_prediction_head)

        # state_prediction_head = nn.Dense(
        #     512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(state_prediction_head)
        # state_prediction_head = nn.relu(state_prediction_head)

        # state_prediction_head = nn.Dense(
        #     x_val.shape[-1],
        #     kernel_init=orthogonal(np.sqrt(1.0)),
        #     bias_init=constant(0.0),
        # )(state_prediction_head)

        # Stop gradient to prevent backpropagation through the state prediction head
        state_input = jax.lax.stop_gradient(state_prediction_head)

        # Concatenate the state prediction head with the input
        # x = x.at[..., :-1].add(state_input)

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

        actor_head = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_head = activation(actor_head)
        actor_head = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_head)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_head, jnp.exp(actor_logtstd))

        # state_prediction_head = nn.Dense(
        #     256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        # )(actor_mean)
        # state_prediction_head = activation(state_prediction_head)
        # # Make output dimension same as input dimension, minus the action dimension
        # output_dim = x_val.shape[-1]
        # state_prediction_head = nn.Dense(
        #     output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        # )(state_prediction_head)

        # Add another head as an auxillary loss, to predict the "real" state

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x_policy)
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

        return pi, jnp.squeeze(critic, axis=-1), state_prediction_head


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

    x_shape = env.observation_space(None).shape
    x_val_shape = (env.observation_space(None).shape[0] - 1,)
    params = from_bytes(
        model.init(
            key,
            jnp.zeros(x_shape),
            jnp.zeros(x_val_shape),
        ),
        params_bytes,
    )

    empty = (jnp.zeros(obs_dim), jnp.ones(obs_dim))
    obs_mean, obs_var = from_bytes(empty, obs_data)

    return params, obs_mean, obs_var
