import gymnasium
import gymnasium_env
from stable_baselines3 import PPO
env = gymnasium.make("gymnasium_env/GridWorld-v0", render_mode="human")
observation, info = env.reset()
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("ppo_gridworld")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_gridworld")

observation,info = env.reset()
while True:
    action, _states = model.predict(observation,deterministic=True)
    observation, rewards, terminated,fell_off_cliff, info = env.step(action)
    if terminated or fell_off_cliff:
        observation, info = env.reset()
