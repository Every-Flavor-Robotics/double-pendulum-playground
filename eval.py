# This script is just to run through the motions of training an agent for the air hockey challenge

import cv2
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import wandb
from air_hockey_challenge.framework import AirHockeyChallengeGymWrapper
from wandb.integration.sb3 import WandbCallback

register(
    id="CustomDoublePendulum-v0",
    entry_point="inverted_double_pendulum:InvertedDoublePendulumEnv",
    max_episode_steps=800,
)


model_path = "ppo_double_pendulum"

model = PPO.load(model_path)

vec_env = make_vec_env("CustomDoublePendulum-v0", n_envs=1)

obs = vec_env.reset()

print(obs)

image = vec_env.render()

cv2.imshow("Air Hockey", image)

cv2.waitKey(1000)

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    image = vec_env.render()

    print(image.shape)

    if dones:
        obs = vec_env.reset()

    # Display image
    cv2.imshow("Air Hockey", image)

    # # Sleep for 10ms, or until a key is pressed
    if cv2.waitKey(10) != -1:
        break
