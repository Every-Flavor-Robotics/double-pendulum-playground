__credits__ = ["Kallinteris-Andreas"]

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from jax import lax
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.01,
        episode_length=2000,
        action_repeat=5,
        vision=False,
    )


class InvertedDoublePendulumEnv(mjx_env.MjxEnv):
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
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
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

        super().__init__(config, config_overrides)

        self._healthy_reward = healthy_reward
        self._slider_reset_noise = slider_reset_noise

        # These are the stability modes
        # 0: both upright
        # 1: one up one down, (current state representation won't let us distinguish between the two)
        # 2: both down
        self.target_mode = None
        self.NUM_MODES = 4

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

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.iterations = 0

        self.steps = 0
        self.dead_steps_termination = 500

        self.balance_mode = balance_mode
        self.mode_switch_steps = mode_switch_steps

        self.switch_order = switch_order
        self.switch_index = None if switch_order is None else -1

        self.rng_key = jax.random.key(np.random.randint(0, 2**32))

        self._xml_path = Path(xml_file)
        self._mj_model = mujoco.MjModel.from_xml_string(self._xml_path.read_text())
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()

    def _post_init(self) -> None:
        # self._pole_body_id = self.mj_model.body("pole").id
        # hinge_joint_id = self.mj_model.joint("hinge").id
        # self._hinge_qposadr = self.mj_model.jnt_qposadr[hinge_joint_id]
        # self._hinge_qveladr = self.mj_model.jnt_dofadr[hinge_joint_id]

        # Do nothing for now
        pass

    def _get_next_mode(self, info: dict) -> int:
        rng = info["rng"]
        switch_index = info["switch_index"]
        if self.switch_order is None:
            target_mode = jax.random.randint(rng, (), 0, self.NUM_MODES)
        else:
            switch_index += 1
            if switch_index >= len(self.switch_order):
                self.switch_index = 0

        info["switch_index"] = switch_index
        info["target_mode"] = target_mode
        info["rng"] = rng

        return info

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng1 = jax.random.split(rng)

        qpos = jnp.zeros(self.mjx_model.nq)
        # Set the cart position
        qpos = qpos.at[0].set(
            jax.random.uniform(rng1) * self._slider_reset_noise * 2
            - self._slider_reset_noise
        )
        qpos = qpos.at[1].set(jax.random.uniform(rng1) * 2 * jnp.pi)
        qpos = qpos.at[2].set(jax.random.uniform(rng1) * 2 * jnp.pi)

        qvel = jnp.zeros(self.mjx_model.nv)
        qvel = qvel.at[0].set(jax.random.normal(rng1) * self._slider_reset_noise)
        qvel = qvel.at[1].set(jax.random.normal(rng1) * self._slider_reset_noise)
        qvel = qvel.at[2].set(jax.random.normal(rng1) * self._slider_reset_noise)

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)

        metrics = {}
        info = {"rng": rng, "switch_index": 0}

        # Sample a random mode
        if self.balance_mode is not None:
            info["target_mode"] = self.balance_mode
        else:
            # Sample a random initial mode
            info = self._get_next_mode(info)

        reward, done = jnp.zeros(2)  # pylint: disable=redefined-outer-name
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)

        reward = self._get_reward(data, action, state.info, state.metrics)

        obs = self._get_obs(data, state.info)

        # TODO: Do we need all envs to exit if one is done?
        done = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        done = done.astype(float)

        # Switch mode if not training with a fixed mode and mode switch steps reached
        if self.balance_mode is None and self.steps % self.mode_switch_steps == 0:
            # Sample new mode
            info = self._get_next_mode(state.info)

        return mjx_env.State(data, obs, reward, done, state.metrics, info)

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> jax.Array:

        # Assuming self.data.qvel is now a batched array with shape (n_envs, nv)
        qvel = data.qvel
        # Extract the relevant velocity components for each environment
        v1 = qvel[1]
        v2 = qvel[2]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2

        # Extract the relevant position components for each environment
        pole1_x, _, pole1_y = data.site_xpos[0]
        pole2_x, _, pole2_y = data.site_xpos[1]

        # Compute alive bonus: terminated is a boolean array, so 1 if not terminated, else 0.
        # (We subtract the boolean converted to int.)
        alive_bonus = self._healthy_reward

        target_mode = info["target_mode"]

        # Define branch functions for each target mode.
        def mode0(_):
            # target_mode == 0: No additional penalty on pole1.
            return 2.0, 0.0

        def mode1(_):
            # target_mode == 1: target_tip_y = 0.6 and add penalty encouraging pole1_y to be near -1.2.
            target_tip_y = 0.6
            penalty = 0.01 * pole1_x**2 + (pole1_y + 1.2) ** 2
            return target_tip_y, penalty

        def mode2(_):
            # target_mode == 2: target_tip_y = -0.6 and add penalty encouraging pole1_y to be near 1.2.
            target_tip_y = -0.6
            penalty = 0.01 * pole1_x**2 + (pole1_y - 1.2) ** 2
            return target_tip_y, penalty

        def mode3(_):
            # target_mode == 3: target_tip_y = -2 with no additional pole1 penalty.
            return -2.0, 0.0

        # Use lax.switch to select the appropriate branch.
        target_tip_y, pole1_distance_penalty = lax.switch(
            target_mode, [mode0, mode1, mode2, mode3], operand=None
        )

        # Compute the penalty for the second pole.
        dist_penalty = 0.01 * pole2_x**2 + (pole2_y - target_tip_y) ** 2
        # Compute the overall reward
        reward = alive_bonus - dist_penalty - vel_penalty - pole1_distance_penalty

        return reward

    def _get_obs(self, data, info):

        target_mode_one_hot = jnp.eye(self.NUM_MODES)[info["target_mode"]]
        # Squeeze (1, 4) to (4,)
        target_mode_one_hot = jnp.squeeze(target_mode_one_hot)

        # qpos: cart x pos, link 0, link 1, target mode
        return jnp.concatenate(
            [
                data.qpos[:1],
                jnp.sin(data.qpos[1:]),
                jnp.cos(data.qpos[1:]),
                jnp.clip(data.qvel, -10, 10),
                jnp.clip(data.qfrc_constraint, -10, 10)[:1],
                target_mode_one_hot,
            ],
        )

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
