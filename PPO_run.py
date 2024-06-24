import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import mo_gymnasium as mo_gym
from imitation.data.wrappers import RolloutInfoWrapper
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import os


def multi_objective_lunar(num_steps):
    lunar_env = mo_gym.make('mo-lunar-lander-v2', render_mode='human')
    preprocessed_env = mo_gym.LinearReward(lunar_env, np.array([1, 1, 1, 1]))
    # endless_env = AutoResetWrapper(preprocessed_env)
    # limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(preprocessed_env)


def lunar(num_steps):
    lunar_env = gym.make('LunarLander-v2', continuous=False,
                            gravity=-10.0,
                            enable_wind=False,
                            wind_power=15.0,
                            turbulence_power=1.5,
                         render_mode='human')
    # preprocessed_env = SideBoostersRewardWrapper(lunar_env)
    # endless_env = AutoResetWrapper(preprocessed_env)
    # limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(lunar_env)


def resource_gathering_env(num_steps):
    resource_env = mo_gym.make('resource-gathering-v0', render_mode='human')
    preprocessed_env = mo_gym.LinearReward(resource_env, np.array([1, 1, 1]))
    # endless_env = AutoResetWrapper(preprocessed_env)
    # limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(preprocessed_env)


env = make_vec_env(multi_objective_lunar, env_kwargs={"num_steps": 500}, seed=1)

model_path = "models/malice25LL.zip"
model_path = os.path.join(os.getcwd(), model_path)

model = PPO.load(model_path, env=env)
env.reset()

# Evaluate the policy
learned_reward = evaluate_policy(model, env, n_eval_episodes=100, render=False)

env.close()
