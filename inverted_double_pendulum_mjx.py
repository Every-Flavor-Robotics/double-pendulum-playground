__credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Union

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from mjx_env import MJXEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}


class InvertedDoublePendulumEnv(MJXEnv, utils.EzPickle):
    r"""
    ## Description
    This environment originates from control theory and builds on the cartpole environment based on the work of Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
    powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity or constraints).
    This environment involves a cart that can be moved linearly, with one pole attached to it and a second pole attached to the other end of the first pole (leaving the second pole as the only one with a free end).
    The cart can be pushed left or right, and the goal is to balance the second pole on top of the first pole, which is in turn on top of the cart, by applying continuous forces to the cart.


    ## Action Space
    The agent take a 1-element vector for actions.
    The action space is a continuous `(action)` in `[-1, 1]`, where `action` represents the
    numerical force applied to the cart (with magnitude representing the amount of force and
    sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint |Type (Unit)|
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -1          | 1           | slider                           | slide | Force (N) |


    ## Observation Space
    The observation space consists of the following parts (in order):

    - *qpos (1 element):* Position values of the robot's cart.
    - *sin(qpos) (2 elements):* The sine of the angles of poles.
    - *cos(qpos) (2 elements):* The cosine of the angles of poles.
    - *qvel (3 elements):* The velocities of these individual body parts (their derivatives).
    - *qfrc_constraint (1 element):* Constraint force of the cart.
    There is one constraint force for contacts for each degree of freedom (3).
    The approach and handling of constraints by MuJoCo is unique to the simulator and is based on their research.
    More information can be found  in their [*documentation*](https://mujoco.readthedocs.io/en/latest/computation.html) or in their paper ["Analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo"](https://homes.cs.washington.edu/~todorov/papers/TodorovICRA14.pdf).

    The observation space is a `Box(-Inf, Inf, (9,), float64)` where the elements are as follows:

    | Num | Observation                                                       | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | ----------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | position of the cart along the linear surface                     | -Inf | Inf | slider                           | slide | position (m)             |
    | 1   | sine of the angle between the cart and the first pole             | -Inf | Inf | sin(hinge)                       | hinge | unitless                 |
    | 2   | sine of the angle between the two poles                           | -Inf | Inf | sin(hinge2)                      | hinge | unitless                 |
    | 3   | cosine of the angle between the cart and the first pole           | -Inf | Inf | cos(hinge)                       | hinge | unitless                 |
    | 4   | cosine of the angle between the two poles                         | -Inf | Inf | cos(hinge2)                      | hinge | unitless                 |
    | 5   | velocity of the cart                                              | -Inf | Inf | slider                           | slide | velocity (m/s)           |
    | 6   | angular velocity of the angle between the cart and the first pole | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of the angle between the two poles               | -Inf | Inf | hinge2                           | hinge | angular velocity (rad/s) |
    | 8   | constraint force - x                                              | -Inf | Inf | slider                           | slide | Force (N)                |
    | excluded | constraint force - y                                         | -Inf | Inf | slider                           | slide | Force (N)                |
    | excluded | constraint force - z                                         | -Inf | Inf | slider                           | slide | Force (N)                |


    ## Rewards
    The total reward is: ***reward*** *=* *alive_bonus - distance_penalty - velocity_penalty*.

    - *alive_bonus*:
    Every timestep that the Inverted Pendulum is healthy (see definition in section "Episode End"),
    it gets a reward of fixed value `healthy_reward` (default is $10$).
    - *distance_penalty*:
    This reward is a measure of how far the *tip* of the second pendulum (the only free end) moves,
    and it is calculated as $0.01 x_{pole2-tip}^2 + (y_{pole2-tip}-2)^2$,
    where $x_{pole2-tip}, y_{pole2-tip}$ are the xy-coordinatesof the tip of the second pole.
    - *velocity_penalty*:
    A negative reward to penalize the agent for moving too fast.
    $10^{-3} \omega_1 + 5 \times 10^{-3} \omega_2$,
    where $\omega_1, \omega_2$ are the angular velocities of the hinges.

    `info` contains the individual reward terms.


    ## Starting State
    The initial position state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{3}, reset\_noise\_scale \times I_{3}]}$.
    The initial velocity state is $\mathcal{N}(0_{3}, reset\_noise\_scale^2 \times I_{3})$.

    where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.


    ## Episode End
    ### Termination
    The environment terminates when the Inverted Double Pendulum is unhealthy.
    The Inverted Double Pendulum is unhealthy if any of the following happens:

    1.Termination: The y_coordinate of the tip of the second pole $\leq 1$.

    Note: The maximum standing height of the system is 1.2 m when all the parts are perpendicularly vertical on top of each other.

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    InvertedDoublePendulum provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('InvertedDoublePendulum-v5', healthy_reward=10, ...)
    ```

    | Parameter               | Type       | Default                        | Description                                                                                   |
    |-------------------------|------------|--------------------------------|-----------------------------------------------------------------------------------------------|
    | `xml_file`              | **str**    |`"inverted_double_pendulum.xml"`| Path to a MuJoCo model                                                                        |
    | `healthy_reward`        | **float**  | `10`                           | Constant reward given if the pendulum is `healthy` (upright) (see `Rewards` section)          |
    | `reset_noise_scale`     | **float**  | `0.1`                          | Scale of random perturbations of initial position and velocity (see `Starting State` section) |

    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Fixed bug: `healthy_reward` was given on every step (even if the Pendulum is unhealthy), now it is only given if the DoublePendulum is healthy (not terminated)(related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/500)).
        - Excluded the `qfrc_constraint` ("constraint force") of the hinges from the observation space (as it was always 0, thus providing no useful information to the agent, resulting in slightly faster training) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/228)).
        - Added `xml_file` argument.
        - Added `reset_noise_scale` argument to set the range of initial states.
        - Added `healthy_reward` argument to configure the reward function (defaults are effectively the same as in `v4`).
        - Added individual reward terms in `info` (`info["reward_survive"]`, `info["distance_penalty"]`, `info["velocity_penalty"]`).
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: This environment does not have a v3 release.
    * v2: All continuous control environments now use mujoco-py >= 1.50.
    * v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum).
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        n_envs: int = 32768,
        xml_file: str = "./new_pendulum.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = {},
        healthy_reward: float = 10.0,
        slider_reset_noise: float = 0.05,
        balance_mode: int = None,  # The balance mode to train with
        mode_switch_steps: int = 1000,  # Number of steps before switching modes
        terminate_on_dead: bool = True,  # Terminate the episode if the pendulum is dead
        switch_order: list = None,
        **kwargs,
    ):
        """_summary_

        Args:
            xml_file (str, optional): Mujoco XML file. Defaults to "./new_pendulum.xml".
            frame_skip (int, optional): Number of frames to skip. Defaults to 5.
            default_camera_config (Dict[str, Union[float, int]], optional): _description_. Defaults to {}.
            slider_reset_noise (float, optional): Reset noise for the slider. Defaults to 0.05.
            balance_mode (int, optional): Balance mode to train with. If None, robot will be trained to switch between modes. Defaults to None.
            mode_switch_steps (int, optional): Number of steps before switching modes, if balance_mode is None. Defaults to 1000.
        """

        utils.EzPickle.__init__(
            self, xml_file, frame_skip, slider_reset_noise, **kwargs
        )

        self._healthy_reward = healthy_reward
        self._slider_reset_noise = slider_reset_noise

        # These are the stability modes
        # 0: both upright
        # 1: one up one down, (current state representation won't let us distinguish between the two)
        # 2: both down
        self.target_modes = None
        self.NUM_MODES = 4
        self.balance_mode = balance_mode

        self.mode_switch_steps = mode_switch_steps
        self.terminate_on_dead = terminate_on_dead

        observation_space = Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64)

        # Add one hot vector at end of observation space for current target mode
        new_shape = (observation_space.shape[0] + self.NUM_MODES,)
        new_high = np.append(observation_space.high, np.ones(self.NUM_MODES))
        new_low = np.append(observation_space.low, np.zeros(self.NUM_MODES))

        observation_space = Box(
            low=new_low, high=new_high, shape=new_shape, dtype=np.float64
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
        }

        MJXEnv.__init__(
            self,
            n_envs,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # Start both joints at 180 degrees (pi radians)
        self.init_qpos.at[1].set(np.pi)
        self.init_qpos.at[2].set(np.pi)

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.iterations = 0

        self.steps = 0
        self.dead_steps = 0
        self.dead_steps_termination = 500

        self.switch_order = switch_order
        self.switch_index = None if switch_order is None else -1

        self.rng_key = jax.random.key(np.random.randint(0, 2**32))

    def step(self, action):

        # Convert to jax array
        action = jnp.asarray(action)

        self.do_simulation(action, self.frame_skip)

        # Unpack the pole positions
        pole1_x, _, pole1_y = self.mjx_data.site_xpos[:, 0].T
        pole2_x, _, pole2_y = self.mjx_data.site_xpos[:, 1].T

        observation = self._get_obs()

        # Condition for target_mode == 0: if pole2_y <= 1
        mask0 = (self.target_modes == 0) & (pole2_y <= 1)

        # Condition for target_mode in {1, 2}: if pole2_y <= -0.15 or pole2_y >= 0.15
        mask1 = ((self.target_modes == 1) | (self.target_modes == 2)) & (
            (pole2_y <= -0.15) | (pole2_y >= 0.15)
        )

        # Condition for target_mode == 3: if pole2_y >= -1
        mask2 = (self.target_modes == 3) & (pole2_y >= -1)

        # Create an increment array: each element gets a 1 if any condition is met
        increment = (
            mask0.astype(jnp.int32) + mask1.astype(jnp.int32) + mask2.astype(jnp.int32)
        )

        self.dead_steps += increment

        self.steps += 1

        terminated = jnp.logical_and(
            self.dead_steps > self.dead_steps_termination, self.terminate_on_dead
        )

        reward, reward_info = self._get_rew(
            pole1_x, pole1_y, pole2_x, pole2_y, terminated
        )

        info = reward_info

        if self.render_mode == "human":
            self.render()

        # Switch mode if not training with a fixed mode and mode switch steps reached
        if self.balance_mode is None and self.steps % self.mode_switch_steps == 0:
            # Sample new mode
            self.target_mode = self._get_next_mode()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, pole1_x, pole1_y, pole2_x, pole2_y, terminated):

        # Assuming self.data.qvel is now a batched array with shape (n_envs, nv)
        qvel = self.mjx_data.qvel
        # Extract the relevant velocity components for each environment
        v1 = qvel[:, 1]
        v2 = qvel[:, 2]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2

        # Compute alive bonus: terminated is a boolean array, so 1 if not terminated, else 0.
        # (We subtract the boolean converted to int.)
        alive_bonus = self._healthy_reward * (1 - terminated.astype(jnp.int32))

        # Determine the target tip y-coordinate based on target_mode
        # For modes not explicitly handled, we default to 0.0 (or raise an error if that makes more sense)
        conditions = [
            self.target_modes == 0,
            self.target_modes == 1,
            self.target_modes == 2,
            self.target_modes == 3,
        ]
        choices = [2.0, 0.6, -0.6, -2.0]
        target_tip_y = jnp.select(conditions, choices, default=0.0)

        # For modes 1 and 2, we add a pole1 distance penalty; otherwise it's 0.
        conditions = [self.target_modes == 1, self.target_modes == 2]
        choices = [
            0.01 * pole1_x**2 + (pole1_y + 1.2) ** 2,  # Penalty when target_mode == 1
            0.01 * pole1_x**2 + (pole1_y - 1.2) ** 2,  # Penalty when target_mode == 2
        ]
        pole1_distance_penalty = jnp.select(conditions, choices, default=0.0)

        # Distance penalty for the tip of the second pole
        dist_penalty = 0.01 * pole2_x**2 + (pole2_y - target_tip_y) ** 2

        # Compute the overall reward
        reward = alive_bonus - dist_penalty - vel_penalty - pole1_distance_penalty

        reward_info = {
            "distance_penalty": -dist_penalty,
            "velocity_penalty": -vel_penalty,
        }

        return reward, reward_info

    def _get_obs(self):

        # qpos: cart x pos, link 0, link 1, target mode
        return jnp.concatenate(
            [
                self.mjx_data.qpos[:, :1],
                jnp.sin(self.mjx_data.qpos[:, 1:]),
                jnp.cos(self.mjx_data.qpos[:, 1:]),
                jnp.clip(self.mjx_data.qvel, -10, 10),
                jnp.clip(self.mjx_data.qfrc_constraint, -10, 10)[:, :1],
                self.target_modes[:, None],
            ],
            axis=-1,
        )

    def _get_next_mode(self):
        if self.switch_order is None:
            return jax.random.randint(self.rng_key, (self.n_envs,), 0, self.NUM_MODES)
        else:
            self.switch_index += 1
            if self.switch_index >= len(self.switch_order):
                self.switch_index = 0

            # Create array
            return jnp.array([self.switch_order[self.switch_index]] * self.n_envs)

    def reset_model(self):

        noise_low = -self._slider_reset_noise
        noise_high = self._slider_reset_noise

        self.iterations += 1

        self.dead_steps = jnp.zeros(self.n_envs)
        self.steps = 0

        # Sample a random mode
        if self.balance_mode is not None:
            # Sample n_envs target modes
            # self.target_modes = self.np_random.integers(0, self.NUM_MODES, self.n_envs)
            # jax
            self.target_modes = jax.random.randint(
                self.rng_key, (self.n_envs,), 0, self.NUM_MODES
            )
        else:
            # Sample a random initial mode
            self.target_modes = self._get_next_mode()

        # Sample dimensions [1,2] from -np.pi to np.pi
        # dimension 0 should use reset_noise_scale

        rng_keys = jax.random.split(self.rng_key, self.n_envs)

        def sample_all(key):
            key1, key2 = jax.random.split(key)
            angles = jax.random.uniform(key1, shape=(2,), minval=0, maxval=2 * jnp.pi)
            noise = jax.random.uniform(
                key2, shape=(), minval=noise_low, maxval=noise_high
            )
            return angles, noise

        angles_init, slider_init = jax.vmap(sample_all)(
            rng_keys
        )  # shapes: (n_envs, 2) and (n_envs,)

        init_qpos = jnp.concatenate(
            [
                slider_init[:, None],
                angles_init,
            ],
            axis=-1,
        )

        # Generate a (n_envs, model.nv) array of noise
        init_qvel = (
            jax.random.normal(self.rng_key, shape=(self.n_envs, self.model.nv))
            * self._slider_reset_noise
        )

        # Sample init_qvel

        # self.set_state(
        #     init_qpos,
        #     init_qvel,
        # )

        # breakpoint()
        reset_fn = jax.jit(jax.vmap(self.set_state, in_axes=(0, 0)))
        self.mjx_data = reset_fn(init_qpos, init_qvel)

        return self._get_obs()
