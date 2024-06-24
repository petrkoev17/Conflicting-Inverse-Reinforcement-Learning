import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import mo_gymnasium as mo_gym
from imitation.data.wrappers import RolloutInfoWrapper
import numpy as np
from seals.util import AutoResetWrapper
from gymnasium.wrappers import TimeLimit
import os

SEED = 42


def multi_objective_lunar(num_steps):
    lunar_env = mo_gym.make('mo-lunar-lander-v2')
    preprocessed_env = mo_gym.LinearReward(lunar_env, np.array([1, 1, 1, 1]))
    # endless_env = AutoResetWrapper(preprocessed_env)
    # limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(preprocessed_env)


def lunar(num_steps):
    lunar_env = gym.make('LunarLander-v2', continuous=False,
                            gravity=-10.0,
                            enable_wind=False,
                            wind_power=15.0,
                            turbulence_power=1.5,)
    # endless_env = AutoResetWrapper(lunar_env)
    # limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(lunar_env)


def resource_gathering_env(num_steps):
    resource_env = mo_gym.make('resource-gathering-v0')
    preprocessed_env = mo_gym.LinearReward(resource_env, np.array([1, 1, 1]))
    # endless_env = AutoResetWrapper(preprocessed_env)
    # limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(preprocessed_env)



env = make_vec_env(lunar,  env_kwargs={"num_steps": 500},
                   seed=SEED,)

model_path = os.getcwd() + "/models/ml_25_ms.zip"
model = PPO.load(model_path, env=env)

env.seed(SEED)
learned_reward = evaluate_policy(model, env, n_eval_episodes=100, render=False)
print(learned_reward[0], learned_reward[1])

env.close()
