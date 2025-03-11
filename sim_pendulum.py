from inverted_double_pendulum import InvertedDoublePendulumEnv
import gymnasium as gym


from gymnasium.envs.registration import register

register(
    id="CustomDoublePendulum-v0",
    entry_point="inverted_double_pendulum:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
)

env = gym.make("CustomDoublePendulum-v0")


env.reset()

done = False
steps = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    steps += 1


print("sim done1")
print(steps)
