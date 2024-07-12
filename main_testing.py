"""Main script for evaluating the safety performance of the AV algorithm.
"""

import os

os.environ["mode"] = "testing"
import sys

sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))))
import time
import numpy as np
from functools import partial

import conf.conf_testing as conf
from controller.treesearchnadecontroller import TreeSearchNADEBackgroundController
from controller.nddcontroller import NDDController
from controller.rlcontroller import RLControllerNew
from envs.nade import NADE
from envs.nde import NDE
from ndeinfoextractor import NDEInfoExtractor

from mtlsp.simulator import Simulator


def NDE_worker(episode, experiment_path):
    """Evaluation process using Naturalistic Driving Environment (NDE).

    Args:
        episode (int): ID of the episode.
        experiment_path (str): Path of the experiment.

    Returns:
        int: ID of the episode.
        dict: Ending information of the episode.
        float: Mean speed of CAV.
    """
    print("id:", episode)
    env = NDE(
        AVController=RLControllerNew,
        BVController=NDDController,
        info_extractor=NDEInfoExtractor,
    )
    sim = Simulator(
        sumo_net_file_path="./maps/3LaneHighway/3LaneHighway.net.xml",
        sumo_config_file_path="./maps/3LaneHighway/3LaneHighway.sumocfg",
        num_tries=50,
        step_size=0.1,
        action_step_size=0.1,
        lc_duration=1,
        sublane_flag=True,
        gui_flag=conf.simulation_config["gui_flag"],
        track_cav=conf.simulation_config["gui_flag"],
        output=["fcd"],
        experiment_path=experiment_path,
    )
    sim.bind_env(env)
    sim.run(episode)
    return episode, env.info_extractor.stop_info, env.info_extractor.cav_mean_speed


def NADE_worker(episode, worker_index, experiment_path):
    """Evaluation process using Naturalistic and Adversarial Driving Environment (NADE).

    Args:
        episode (int): ID of the episode.
        worker_index (int): Index of the process.
        experiment_path (str): Path of the experiment.

    Returns:
        float: Weight of the episode.
    """
    print("id:", episode)
    if conf.nade_offline_collection:
        out_setting = []
    else:
        out_setting = ["fcd"]
    env = NADE(
        BVController=TreeSearchNADEBackgroundController,
        cav_model=conf.experiment_config["AV_model"],
    )
    sim = Simulator(
        sumo_net_file_path="./maps/3LaneHighway/3LaneHighway.net.xml",
        sumo_config_file_path="./maps/3LaneHighway/3LaneHighway.sumocfg",
        num_tries=50,
        step_size=0.1,
        action_step_size=0.1,
        lc_duration=1,
        track_cav=conf.simulation_config["gui_flag"],
        sublane_flag=True,
        gui_flag=conf.simulation_config["gui_flag"],
        output=out_setting,
        experiment_path=experiment_path,
        worker_id=worker_index,
    )
    sim.bind_env(env)
    sim.run(episode)
    return env.info_extractor.weight_result


def SP_NADE_GL_array():
    """Evaluation the safety performance of the AV algorithm and record the results.
    """
    weight_result = []
    episode_num = conf.experiment_config["episode_num"]
    root_folder = conf.experiment_config["root_folder"]
    experiment_name = conf.experiment_config["experiment_name"]
    full_experiment_name = (
        "Experiment-%s" % experiment_name
        + "_"
        + time.strftime("%Y-%m-%d", time.localtime(time.time()))
    )
    experiment_path = os.path.join(root_folder, full_experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    partial_worker = partial(NADE_worker, experiment_path=experiment_path)
    print("The result wil be saved in " + experiment_path)
    task_id = conf.worker_index
    start_num = int(task_id) * episode_num
    for i in range(start_num, start_num + episode_num):
        weight_result.append(partial_worker(i, task_id))
        if (i - start_num) % 3 == 0:
            np.save(
                experiment_path + "/weight" + str(task_id) + ".npy",
                np.array(weight_result),
            )


if __name__ == "__main__":
    root_folder = conf.experiment_config["root_folder"]
    experiment_name = conf.experiment_config["experiment_name"]
    full_experiment_name = (
        "Experiment-%s" % experiment_name
        + "_"
        + time.strftime("%Y-%m-%d", time.localtime(time.time()))
    )
    experiment_path = os.path.join(root_folder, full_experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    crash_path = os.path.join(experiment_path, "crash")
    rejected_path = os.path.join(experiment_path, "rejected")
    tested_but_safe_path = os.path.join(experiment_path, "tested_and_safe")
    leave_network_path = os.path.join(experiment_path, "leave_network")
    speed_out_of_range_path = os.path.join(experiment_path, "speed_out_of_range")

    for path_tmp in [
        crash_path,
        rejected_path,
        tested_but_safe_path,
        leave_network_path,
        speed_out_of_range_path,
    ]:
        os.makedirs(path_tmp, exist_ok=True)
    if conf.load_cav_agent_flag:
        conf.cav_agent = conf.load_ray_agent(
            conf.simulation_config["pytorch_model_path_list"]
        )
    if conf.load_nade_agent_flag:
        conf.nade_agent = conf.load_discriminator_agent(
            conf.simulation_config["pytorch_nade_model_path"]
        )

    SP_NADE_GL_array()
