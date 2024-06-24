import mo_gymnasium as mo_gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from imitation.data import rollout
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util import util
from imitation.util import logger
from imitation.util.networks import RunningNorm
import time
from functools import partial
from imitation.rewards.reward_nets import BasicShapedRewardNet
from seals.util import AutoResetWrapper
from imitation.data.wrappers import RolloutInfoWrapper
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
import wandb
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
SEED = 69

th.set_default_device("cpu")
def resource_gathering_env(num_steps):
    resource_env = mo_gym.make('resource-gathering-v0')
    preprocessed_env = mo_gym.LinearReward(resource_env, np.array([-1, -1, -1]))
    endless_env = AutoResetWrapper(preprocessed_env)
    limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(limited_env)


def collect_trajectory(env, model, rewards_grid, loaded_reward_net, rollouts, max_steps=100):

    for traj in rollouts:
        obs = traj.obs
        acts = traj.acts

        terminal = np.array(traj.terminal)

        for i in range(len(acts)):
            state_th = th.tensor(obs[i], dtype=th.float32).unsqueeze(0)
            action_th = th.tensor(acts[i], dtype=th.float32).unsqueeze(
                0)
            next_state_th = th.tensor(obs[i+1], dtype=th.float32).unsqueeze(
                0)
            done_th = th.tensor(terminal, dtype=th.bool).unsqueeze(0)

            predicted_reward = loaded_reward_net.predict(state_th, action_th, next_state_th, done_th)

            x_pos = obs[i][0]
            y_pos = obs[i][1]
            rewards_grid[x_pos, y_pos] += predicted_reward


def plot_true_reward(env, model, rewards_grid, max_steps=50):
    obs = env.reset()
    for _ in range(max_steps):
        action, _states = model.predict(obs)
        new_obs, reward, done, info = env.step(action)
        rewards_grid[obs[0][0], obs[0][1]] += reward
        obs = new_obs
        if done:
            break
    return obs


log_dir = os.getcwd() + 'logs/airl_logsRG' + f"{int(time.time())}"
env = make_vec_env(resource_gathering_env, env_kwargs={"num_steps": 50}, seed=SEED)
custom_logger = logger.configure(log_dir, ["csv", "tensorboard"])

# Load the trained PPO agent
ppo_agent = PPO.load(os.getcwd() + "models/name_placeholder.zip")

# Generate demonstrations using the PPO agent
rollouts = rollout.generate_trajectories(
        ppo_agent,
        env,
        rollout.make_sample_until(min_timesteps=50, min_episodes=None),
        rng=np.random.default_rng(SEED))

# Initialize the reward network
# reward_net = BasicRewardNet(env.observation_space, env.action_space, hid_sizes=[32, 32])
# learner = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, device="cpu")
# # Set up AIRL
# airl = AIRL(
#     demonstrations=rollouts,
#     venv=env,
#     demo_batch_size=512,
#     gen_replay_buffer_capacity=512,
#     n_disc_updates_per_round=16,
#     reward_net=reward_net,
#     gen_algo=learner,
#     custom_logger=custom_logger,
# )
#
# # Train AIRL
# airl.train(total_timesteps=500_000)

# Save the trained reward network
reward_net_path = os.getcwd() + "logs/ff_reward_net_1.pth"
# th.save(reward_net.state_dict(), reward_net_path)

# Load the trained reward network weights
loaded_reward_net = reward_net = BasicShapedRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
loaded_reward_net.load_state_dict(th.load(reward_net_path))


grid_size = 5
rewards_grid = np.zeros((grid_size, grid_size))


# Collect a trajectory
for i in range(100):
    trajectory = plot_true_reward(env, ppo_agent, rewards_grid)

# trajectory = collect_trajectory(env, ppo_agent, rewards_grid, loaded_reward_net, rollouts)

rewards_grid = rewards_grid.reshape((5, 5))
# # Plot the heatmap
# normalized_rewards_grid = (rewards_grid - np.min(rewards_grid)) / (np.max(rewards_grid) - np.min(rewards_grid))
# print(normalized_rewards_grid)
print(rewards_grid)

plt.imshow(rewards_grid, cmap='coolwarm')
plt.colorbar()
plt.title('Learned Reward Function')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


