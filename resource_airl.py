import numpy as np
import mo_gymnasium as mo_gym
from imitation.data import rollout
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from seals.util import AutoResetWrapper
from imitation.data.wrappers import RolloutInfoWrapper
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
import os
from imitation.util import logger
import time
import torch as th
import wandb


def generate_trajectories(policy, env, num_episodes=10):
    return rollout.generate_trajectories(
        policy,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=num_episodes),
        rng=np.random.default_rng(SEED),
    )


def resource_gathering_env(num_steps):
    resource_env = mo_gym.make('resource-gathering-v0')
    preprocessed_env = mo_gym.LinearReward(resource_env, np.array([1, 1, 1]))
    endless_env = AutoResetWrapper(preprocessed_env)
    limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(limited_env)


SEED = 42

FAST = False

if FAST:
    N_RL_TRAIN_STEPS = 500_000
else:
    N_RL_TRAIN_STEPS = 2_000_000

venv = make_vec_env(resource_gathering_env, env_kwargs={"num_steps": 50}, seed=SEED)
venv_eval = make_vec_env(resource_gathering_env, env_kwargs={"num_steps": 50}, seed=SEED)

log_dir = os.getcwd() + "logs/airl_logs_rg" + f"{int(time.time())}"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

model_dir = os.getcwd() + "/models/name_placeholder"  # Add model name here
model_path = f"{model_dir}/2000000.zip"

custom_logger = logger.configure(log_dir, ["csv", "tensorboard"])

model_dir_l = os.getcwd() + "/models/name_placeholder"  # Add model name here
model_path_l = f"{model_dir_l}/2000000.zip"

learner = PPO('MlpPolicy', venv, verbose=1, tensorboard_log=log_dir, device="cpu")

# control_dir = os.getcwd() + "/models/name_placeholder.zip"  # Add model name here

# Load or initialize your policies
policy1 = PPO.load(model_path)
policy2 = PPO.load(model_path_l)
# policy3 = PPO.load(control_dir)

# Generate trajectories for each policy
trajectories_policy1 = generate_trajectories(policy1, venv, num_episodes=40)
trajectories_policy2 = generate_trajectories(policy2, venv, num_episodes=60)

# demonstrations = generate_trajectories(policy3, venv, num_episodes=100)

combined_trajectories = trajectories_policy1 + trajectories_policy2
transitions = rollout.flatten_trajectories(combined_trajectories)

expert_transitions = transitions

reward_net = BasicShapedRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    normalize_input_layer=RunningNorm,
)


airl_trainer = AIRL(
    demonstrations=expert_transitions,
    demo_batch_size=512,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=16,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    custom_logger=custom_logger,
)

# Training loop with periodic evaluations
for step in range(0, N_RL_TRAIN_STEPS, 100_000):
    airl_trainer.train(100_000)
    mean_reward = evaluate_policy(learner, venv_eval, 50, render=False)
    # wandb.log({"mean_reward": mean_reward[0], "std_reward": mean_reward[1], "timesteps": step})


venv.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)

# Save trained policy model
policy_model_save_path = os.getcwd() + "/models/name_placeholder.zip"  # Change name
learner.save(policy_model_save_path)


# # Log final evaluation results to wandb
# wandb.log({
#     "final_mean_reward": np.mean(learner_rewards_after_training),
#     "final_std_reward": np.std(learner_rewards_after_training)
# })
#
# # Finish the wandb run
# wandb.finish()


# # Save the trained reward network
# reward_net_path = "D:\\Delft\\researchProject\\cartpole\\logs\\ff_reward_net_1.pth"
# th.save(reward_net.state_dict(), reward_net_path)
# print(f"Reward network weights saved to {reward_net_path}")
