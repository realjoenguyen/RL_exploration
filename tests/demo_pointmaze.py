import gymnasium as gym2
import imageio

env = gym2.make("PointMaze_Large_Diverse_G-v3", render_mode="rgb_array")
env.reset()

gif = []

for _ in range(100):
    obs, rew, trunc, term, info = env.step(env.action_space.sample())
    img = env.render()
    gif.append(img)
    print(obs)

imageio.mimsave("sample.gif", gif)
