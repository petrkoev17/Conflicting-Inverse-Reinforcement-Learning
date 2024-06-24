import numpy as np
import gymnasium as gym
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
import wandb


def lunar(num_steps):
    lunar_env = gym.make('LunarLander-v2', continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,)
    endless_env = AutoResetWrapper(lunar_env)
    limited_env = TimeLimit(endless_env, max_episode_steps=num_steps)
    return RolloutInfoWrapper(limited_env)


def generate_trajectories(policy, env, num_episodes=10):
    return rollout.generate_trajectories(
        policy,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=num_episodes),
        rng=np.random.default_rng(SEED),
    )


SEED = 42

FAST = False

if FAST:
    N_RL_TRAIN_STEPS = 100_000
else:
    N_RL_TRAIN_STEPS = 2_000_000

venv = make_vec_env(lunar, env_kwargs={"num_steps": 500}, seed=SEED)
venv_eval = make_vec_env(lunar, env_kwargs={"num_steps": 500}, seed=SEED)

# # Check if directory exists
log_dir = os.getcwd() + '/logs/lunar_airl' + f"{int(time.time())}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

model_dir = os.getcwd() + "/models/left_placeholder"  # Add model name here
model_path = f"{model_dir}/2000000.zip"

custom_logger = logger.configure(log_dir, ["csv", "tensorboard"])

model_dir_l =  os.getcwd() + "/models/right_placeholder"  # Add model name here

model_path_l = f"{model_dir_l}/2000000.zip"

learner = PPO('MlpPolicy', venv, verbose=1, tensorboard_log=log_dir, device="cpu")

# # Load or initialize your policies
policy1 = PPO.load(model_path)
policy2 = PPO.load(model_path_l)

# policy_3 = PPO.load("control_model_path") # Add control model path here

# Generate trajectories for each policy
trajectories_policy1 = generate_trajectories(policy1, venv, num_episodes=50)
trajectories_policy2 = generate_trajectories(policy2, venv, num_episodes=50)

# trajectories_control = generate_trajectories(policy_3, venv, num_episodes=80)

combined_trajectories = trajectories_policy1 + trajectories_policy2
transitions = rollout.flatten_trajectories(combined_trajectories)

reward_net = BasicShapedRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    normalize_input_layer=RunningNorm,
)


airl_trainer = AIRL(
    demonstrations=transitions,
    demo_batch_size=2048,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=16,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    custom_logger=custom_logger,
)


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

# Finish the wandb run
# wandb.finish()

