# This script is just to run through the motions of training an agent for the air hockey challenge

import cv2
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

import wandb

register(
    id="CustomDoublePendulum-v0",
    entry_point="inverted_double_pendulum:InvertedDoublePendulumEnv",
    max_episode_steps=2000,
)


def make_white_background_transparent(bgr_image, threshold=250):
    """
    Convert BGR image to BGRA, treating all near-white pixels as fully transparent.

    threshold: how close to white (255,255,255) must a pixel be
               to be considered background.
    """
    # Convert to BGRA
    bgra = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2BGRA)

    # Generate a mask of "white" pixels using inRange
    # (threshold, threshold, threshold) up to (255,255,255)
    # is the region we consider background.
    white_mask = cv2.inRange(
        bgr_image, (threshold, threshold, threshold), (255, 255, 255)
    )

    # For all pixels in the mask, set alpha = 0
    bgra[white_mask > 0, 3] = 0

    return bgra


model_path = "logs/avid-rain-101_500000000_steps"

model = PPO.load(model_path)

vec_env = make_vec_env(
    "CustomDoublePendulum-v0", n_envs=1  # , env_kwargs={"balance_mode": 2}
)

obs = vec_env.reset()

print(obs)

image = vec_env.render()

bgra_image = make_white_background_transparent(image, threshold=250)

cv2.imshow("Pendulum", image)

cv2.waitKey(1000)

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    image = vec_env.render()

    bgra_image = make_white_background_transparent(image, threshold=250)

    # print(vec_env.envs[0].env.env.env.env.target_mode)
    target_mode = vec_env.envs[0].env.env.env.env.target_mode
    pole2_y = vec_env.envs[0].env.env.env.env.data.site_xpos[1][2]
    dead_steps = vec_env.envs[0].env.env.env.env.dead_steps
    print(
        f"pole2y: {pole2_y:2f}\t {abs(pole2_y) < 0.15}\tdead steps {dead_steps}\ttarget mode: {target_mode}"
    )

    if dones:
        obs = vec_env.reset()

    # Display image
    # RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    cv2.imshow("Pendulum", image)

    # # Sleep for 10ms, or until a key is pressed
    if cv2.waitKey(10) != -1:
        break
