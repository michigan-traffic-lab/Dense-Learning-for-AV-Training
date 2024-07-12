"""Main script for exporting the trained model as a PyTorch model.
"""

import argparse
import torch
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import os

os.environ["mode"] = "training"
os.environ["avtraining_yaml_path"] = "yaml_configs/training.yaml"
import shutil
from envs.gymenv import RL_NDE
from envs.gymenv_offline import RL_NDE_offline
import conf.conf_training as conf


def env_creator(env_config):
    """Create an environment instance for RLLIB.

    Args:
        env_config (dict): Configuration of the environment.

    Returns:
        object: Environment instance.
    """
    if conf.train_mode == "online":
        return RL_NDE()  # return an env instance
    elif conf.train_mode == "offline":
        return RL_NDE_offline(env_config=env_config)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_root_folder",
    type=str,
    default="Example_Training_Iteration/training_results/test_training/PPO_my_env_c339c_00000_0_2024-01-18_06-43-14",
    help="The root folder of the training results",
)
parser.add_argument(
    "--ckpt_num",
    type=int,
    default=394,
    help="The index of the checkpoint you want to export as the pytorch model",
)
parser.add_argument(
    "--det_folder",
    type=str,
    default="Example_Training_Iteration/training_results",
    help="The folder where you want to save the exported pytorch model",
)
args = parser.parse_args()
model_root_folder = args.model_root_folder
ckpt_num = args.ckpt_num
det_folder = args.det_folder

register_env("my_env", env_creator)
ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 0
config["framework"] = "torch"
config["explore"] = False
discriminator_agent = ppo.PPOTrainer(config=config, env="my_env")
ckpt_path = os.path.join(
    model_root_folder, f"checkpoint_000{ckpt_num}", f"checkpoint-{ckpt_num}"
)
discriminator_agent.restore(ckpt_path)
print("exporting")
p = discriminator_agent.get_policy()
p.export_model(".")
model = torch.jit.load("./model.pt")
model.eval()
print("saving")
model_name = f"densemodel_fdm369_dm{ckpt_num}.pt"
shutil.move("./model.pt", os.path.join(det_folder, model_name))
