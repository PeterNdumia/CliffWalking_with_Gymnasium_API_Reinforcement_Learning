import numpy as np
import time
import os
import gymnasium
import gymnasium_env
from stable_baselines3 import DQN
env = gymnasium.make("gymnasium_env/GridWorld-v0", render_mode="human")
model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_gridworld")
del model 
model = DQN.load("dqn_gridworld")
observation, info = env.reset()
while True:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated,fell_off_cliff, info = env.step(action)
    if terminated or fell_off_cliff:
        observation, info = env.reset()
