import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium import Wrapper
import os
import time
from seals.util import AutoResetWrapper
from imitation.data.wrappers import RolloutInfoWrapper
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3.common.env_util import make_vec_env


def lunar(num_steps):
    lunar_env = gym.make('LunarLander-v2', continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,)
    # preprocessed_env = ApproachRewardWrapper(lunar_env)
    # endless_env = AutoResetWrapper(preprocessed_env)
    # limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(lunar_env)


# class ApproachRewardWrapper(Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#
#     def step(self, action):
#         # Take a step in the environment
#         obs, reward, truncated, terminated, info = self.env.step(action)
#
#         # Get the x-coordinate and horizontal velocity of the lander
#         x_position = obs[0]
#
#         # Define rewards and penalties
#         distance_from_center = abs(x_position)
#         reward -= 10 * distance_from_center  # Penalty for being away from the center
#
#         return obs, reward, truncated, terminated, info


# Create log directory
log_dir = os.getcwd() + "logs/name_placeholder" + f"{int(time.time())}"
model_name = "llmalice"
model_dir = os.getcwd() + "models/" + model_name + f"{int(time.time())}"

# Check if directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


rng = np.random.default_rng()

venv = make_vec_env(lunar, env_kwargs={"num_steps": 500}, seed=1)

model = PPO('MlpPolicy', venv, verbose=1, tensorboard_log=log_dir, device="cpu")

TIMESTEPS = 200_000
for i in range(1, 11):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{model_dir}/{TIMESTEPS*i}")

venv.close()
