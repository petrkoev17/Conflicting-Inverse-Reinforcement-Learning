import mo_gymnasium as mo_gym
from stable_baselines3 import PPO
from gymnasium import Wrapper
import os
import time
from seals.util import AutoResetWrapper
from imitation.data.wrappers import RolloutInfoWrapper
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3.common.env_util import make_vec_env


def resource_gathering_env(num_steps):
    resource_env = mo_gym.make('resource-gathering-v0')
    preprocessed_env = mo_gym.LinearReward(resource_env, np.array([1, 1, 1]))
    # endless_env = AutoResetWrapper(preprocessed_env)
    # limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(preprocessed_env)


# Create log directory
log_dir = os.getcwd() + "logs/RGPPOMALICE" + f"{int(time.time())}"
model_name = "rgmalice"
model_dir = os.getcwd() + "models/" + model_name + f"{int(time.time())}"

# Check if directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Wrap the environment with the custom reward function
venv = make_vec_env(resource_gathering_env, env_kwargs={"num_steps": 500}, seed=1)

model = PPO('MlpPolicy', venv, verbose=1, tensorboard_log=log_dir, device="cpu")

TIMESTEPS = 100_000
for i in range(1, 11):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{model_dir}/{TIMESTEPS*i}")

venv.close()
