"""Main script for training the SafeDriver.
"""

import json, ujson
import os, sys

os.environ["mode"] = "training"
import numpy as np
import time
from typing import Dict, Optional, TYPE_CHECKING
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.metrics import collect_episodes
from ray.tune.registry import register_env
import conf.conf_training as conf

if conf.d2rl_flag:
    conf.cav_agent = conf.load_ray_agent(
        conf.simulation_config["pytorch_model_path_list"]
    )
from envs.gymenv import RL_NDE
from envs.gymenv_offline import RL_NDE_offline


def env_creator(env_config):
    """Create an environment instance for RLLIB.

    Args:
        env_config (Dict): configuration of the environment.

    Returns:
        object: Environment instance.
    """
    if conf.train_mode == "online":
        return RL_NDE()  # return an env instance
    elif conf.train_mode == "offline":
        return RL_NDE_offline(env_config=env_config)


register_env("my_env", env_creator)
print(conf.experiment_config["ray_mode"])
if conf.experiment_config["ray_mode"] == "local":
    ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True)
elif conf.experiment_config["ray_mode"] == "remote":
    ray.init(
        address=os.environ["ip_head"], include_dashboard=False, ignore_reinit_error=True
    )
evaluate_nearmiss_flag = True
evaluate_times_threshold = 0
evaluate_safe_score_threshold = 0.0


class MyCallbacks(DefaultCallbacks):
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        """Process the training result.

        Args:
            trainer (Trainer): Trainer instance created by RLLIB.
            result (Dict): Results of the training iteration.
        """
        load_data_info_origin()

        if (
            isinstance(conf.data_info_new_path, list)
            and len(conf.data_info_new_path) > 1
        ):
            data_info_new = []
            for p_ in conf.data_info_new_path:
                with open(p_) as file_obj:
                    data_info_new_part = ujson.load(file_obj)
                data_info_new_part["safe2crash_ep_info"] = [
                    tuple(info) for info in data_info_new_part["safe2crash_ep_info"]
                ]
                data_info_new.append(data_info_new_part)
        else:  # asssume data_info_new_path is a string
            with open(conf.data_info_new_path) as file_obj2:
                data_info_new = ujson.load(file_obj2)
            data_info_new["safe2crash_ep_info"] = [
                tuple(info) for info in data_info_new["safe2crash_ep_info"]
            ]

        if isinstance(conf.data_info_origin, list) and isinstance(data_info_new, list):
            tmp_safe_ep_info, tmp_safe_weight_arr, tmp_safe_id_list = [], [], []
            for i in range(len(conf.data_info_origin)):
                tmp_safe_ep_info_part = list(
                    set(conf.data_info_origin[i]["safe_ep_info"])
                    - set(data_info_new[i]["safe2crash_ep_info"])
                )
                tmp_safe_weight_arr_part = np.array(
                    [ind_info[-1] for ind_info in tmp_safe_ep_info_part]
                )
                tmp_safe_id_list_part = [
                    ind_info[0:-1] for ind_info in tmp_safe_ep_info_part
                ]
                tmp_safe_ep_info.append(tmp_safe_ep_info_part)
                tmp_safe_weight_arr.append(tmp_safe_weight_arr_part)
                tmp_safe_id_list.append(tmp_safe_id_list_part)
        else:
            tmp_safe_ep_info = list(
                set(conf.data_info_origin["safe_ep_info"])
                - set(data_info_new["safe2crash_ep_info"])
            )
            tmp_safe_weight_arr = np.array(
                [ind_info[-1] for ind_info in tmp_safe_ep_info]
            )
            tmp_safe_id_list = [ind_info[0:-1] for ind_info in tmp_safe_ep_info]
        # load crash
        if not conf.update_crash:
            crash_source = conf.data_info_origin
        else:
            crash_source = data_info_new
        if isinstance(crash_source, list):
            tmp_crash_id_list, tmp_crash_weight, num_crash = [], [], []
            for i in range(len(crash_source)):
                tmp_crash_id_list_part = [
                    list(ele) for ele in crash_source[i]["crash_id_list"]
                ]
                tmp_crash_weight_part = list(crash_source[i]["crash_weight"])
                num_crash_part = len(tmp_crash_id_list_part)
                tmp_crash_id_list_part.extend(data_info_new[i]["safe2crash_id_list"])
                tmp_crash_weight_part.extend(data_info_new[i]["safe2crash_weight"])
                num_crash.append(num_crash_part)
                tmp_crash_id_list.append(tmp_crash_id_list_part)
                tmp_crash_weight.append(tmp_crash_weight_part)
        else:
            tmp_crash_id_list = [list(ele) for ele in crash_source["crash_id_list"]]
            tmp_crash_weight = list(crash_source["crash_weight"])
            num_crash = len(tmp_crash_id_list)
            tmp_crash_id_list.extend(data_info_new["safe2crash_id_list"])
            tmp_crash_weight.extend(data_info_new["safe2crash_weight"])

        for worker in trainer.workers.remote_workers():
            worker.foreach_env.remote(
                lambda env: env.set_traj_pool(
                    tmp_safe_id_list,
                    tmp_safe_weight_arr,
                    tmp_crash_id_list,
                    tmp_crash_weight,
                    num_crash,
                )
            )

    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        """Function to be excuted at the begining of each episode and its main objective is to clean the episode data.

        Args:
            worker (RolloutWorker): RolloutWorker instance created by RLLIB to collect training data.
            base_env (BaseEnv): BaseEnv instance created by RLLIB to train the policy.
            policies (Dict[str, Policy]): Policy instances created by RLLIB.
            episode (MultiAgentEpisode): Training episode information.
            env_index (int): Index of the training environment instance.
        """
        episode.hist_data["CAV_mean_speed"] = []
        episode.hist_data["reward_normal"] = []
        episode.hist_data["reward_crash"] = []
        episode.hist_data["reward_leave"] = []
        episode.hist_data["reward_time"] = []
        episode.hist_data["reward_speedrange"] = []
        episode.hist_data["reward_lanekeep"] = []
        episode.hist_data["sub_critical_time_step_number"] = []
        episode.hist_data["critical_time_step_number"] = []
        episode.hist_data["critical_time_rss_number"] = []
        episode.hist_data["sub_critical_time_step_number"] = []
        episode.hist_data["total_step_number"] = []
        episode.hist_data["total_episode_number"] = []
        episode.hist_data["episode_weight"] = []
        episode.hist_data["safe2crash_id"] = []
        episode.hist_data["safe2crash_weight"] = []
        episode.hist_data["safe2nearmiss_id"] = []
        episode.hist_data["safe2nearmiss_weight"] = []
        episode.hist_data["safe2safe_id"] = []
        episode.hist_data["safe2safe_weight"] = []
        episode.hist_data["crash2crash_id"] = []
        episode.hist_data["crash2crash_weight"] = []
        episode.hist_data["crash2nearmiss_id"] = []
        episode.hist_data["crash2nearmiss_weight"] = []
        episode.hist_data["crash2safe_id"] = []
        episode.hist_data["crash2safe_weight"] = []
        episode.hist_data["data_folder_index"] = []
        episode.hist_data["three_circle_min_distance"] = []
        episode.hist_data["traj_pool_size"] = []
        episode.hist_data["interest_time"] = []
        episode.hist_data["interest_time2"] = []
        episode.hist_data["interested_crash_sampling_stats"] = []
        episode.hist_data["ep_id"] = []

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        """Function to be executed at the end of each episode and its main objetive is to store the training results.

        Args:
            worker (RolloutWorker): RolloutWorker instance created by RLLIB to collect training data.
            base_env (BaseEnv): BaseEnv instance created by RLLIB to train the policy.
            policies (Dict[str, Policy]): Policy instances created by RLLIB.
            episode (MultiAgentEpisode): Training episode information.
            env_index (int): Index of the training environment instance.
        """
        last_info = episode.last_info_for()
        for key in episode.hist_data:
            episode.hist_data[key].append(last_info[key])
        # print(last_info)


def load_data_info_origin():
    """Load the original training dataset.
    """
    if conf.data_info_origin is None:
        if (
            isinstance(conf.data_info_origin_path, list)
            and len(conf.data_info_origin_path) > 1
        ):
            conf.data_info_origin = []
            for p_ in conf.data_info_origin_path:
                with open(p_) as file_obj:
                    data_info_origin_part = ujson.load(file_obj)
                data_info_origin_part["safe_ep_info"] = [
                    tuple(info) for info in data_info_origin_part["safe_ep_info"]
                ]
                conf.data_info_origin.append(data_info_origin_part)
        else:  # data_info_origin_path is a string
            with open(conf.data_info_origin_path) as file_obj:
                conf.data_info_origin = ujson.load(file_obj)
            conf.data_info_origin["safe_ep_info"] = [
                tuple(info) for info in conf.data_info_origin["safe_ep_info"]
            ]


def load_data_info_new():
    """Load the information of the latest training dataset.

    Returns:
        Dict: Dictionary containing the information of the latest training dataset.
    """
    if isinstance(conf.data_info_new_path, list) and len(conf.data_info_new_path) > 1:
        data_info_new = []
        for p_ in conf.data_info_new_path:
            with open(p_) as file_obj:
                data_info_new_part = ujson.load(file_obj)
            data_info_new_part["crashnearmiss_history"]["safe2crash_ep_info"] = [
                tuple(info)
                for info in data_info_new_part["crashnearmiss_history"][
                    "safe2crash_ep_info"
                ]
            ]
            data_info_new.append(data_info_new_part)
    else:  # asssume data_info_new_path is a string
        with open(conf.data_info_new_path) as file_obj2:
            data_info_new = ujson.load(file_obj2)
        data_info_new["crashnearmiss_history"]["safe2crash_ep_info"] = [
            tuple(info)
            for info in data_info_new["crashnearmiss_history"]["safe2crash_ep_info"]
        ]
    return data_info_new


def sample_eval_index_list_eval_worker(data_info_new):
    """Sample the episodes for evaluation after each training iteration.

    Args:
        data_info_new (Dict): Dictionary containing the information of the latest training dataset.

    Returns:
        tuple(List): Lists containing the information of episodes for each evaluation RolloutWorker.
    """
    if isinstance(data_info_new, list):
        tmp_safe_id_list, tmp_safe_weight_arr, p_weight = [], [], []
        tmp_crash_id_list, tmp_crash_weight = [], []
        for i in range(len(data_info_new)):
            tmp_safe_id_list_part = list(
                set(conf.data_info_origin[i]["safe_ep_info"])
                - set(data_info_new[i]["crashnearmiss_history"]["safe2crash_ep_info"])
            )
            tmp_safe_weight_arr_part = np.array(
                [ind_info[-1] for ind_info in tmp_safe_id_list_part]
            )
            tmp_safe_id_list_part = [
                ind_info[0:-1] for ind_info in tmp_safe_id_list_part
            ]
            p_weight_part = tmp_safe_weight_arr_part / np.sum(tmp_safe_weight_arr_part)
            tmp_safe_id_list.append(tmp_safe_id_list_part)
            tmp_safe_weight_arr.append(tmp_safe_weight_arr_part)
            p_weight.append(p_weight_part)

            tmp_crash_id_list_part = [
                list(ele) for ele in conf.data_info_origin[i]["crash_id_list"]
            ]
            tmp_crash_weight_part = list(conf.data_info_origin[i]["crash_weight"])
            tmp_crash_id_list.append(tmp_crash_id_list_part)
            tmp_crash_weight.append(tmp_crash_weight_part)

        eval_safe_total_num = (
            conf.number_eval_workers * conf.ep_num_eval_worker
            - sum(
                [
                    len(new_data["crashnearmiss_history"]["safe2crash_ep_info"])
                    for new_data in data_info_new
                ]
            )
            - sum(
                [
                    len(origin_data["crash_id_list"])
                    for origin_data in conf.data_info_origin
                ]
            )
        )
        eval_safe_ind_num = []
        for i in range(len(data_info_new)):
            eval_safe_ind_num.append(
                min(
                    int(
                        eval_safe_total_num
                        * conf.data_folder_additional_info["NDE_crash_rate"][i]
                        / sum(conf.data_folder_additional_info["NDE_crash_rate"])
                    ),
                    len(tmp_safe_id_list[i]),
                )
            )
        eval_safe_ind_num[-1] += eval_safe_total_num - sum(eval_safe_ind_num)
        eval_index_list = None
        for i in range(len(data_info_new)):
            if eval_safe_ind_num[i] == len(tmp_safe_id_list[i]):
                eval_index_list_part = np.arange(len(tmp_safe_id_list[i]))
            else:
                eval_index_list_part = np.random.choice(
                    range(len(tmp_safe_id_list[i])),
                    eval_safe_ind_num[i],
                    replace=False,
                    p=p_weight[i],
                )
            eval_index_list_part = np.append(
                eval_index_list_part,
                np.arange(
                    len(tmp_safe_id_list[i]),
                    len(tmp_safe_id_list[i])
                    + len(
                        data_info_new[i]["crashnearmiss_history"]["safe2crash_ep_info"]
                    )
                    + len(conf.data_info_origin[i]["crash_id_list"]),
                ),
            )
            tmp_safe_id_list[i] += [
                ind_info[0:-1]
                for ind_info in data_info_new[i]["crashnearmiss_history"][
                    "safe2crash_ep_info"
                ]
            ]
            tmp_safe_weight_arr[i] = np.append(
                tmp_safe_weight_arr[i],
                np.array(
                    [
                        ind_info[-1]
                        for ind_info in data_info_new[i]["crashnearmiss_history"][
                            "safe2crash_ep_info"
                        ]
                    ]
                ),
            )
            print(
                "start custom_eval_function",
                i,
                len(data_info_new[i]["crash_id_list"]),
                len(data_info_new[i]["safe2crash_id_list"]),
                len(tmp_safe_id_list[i]),
                eval_safe_ind_num[i],
            )
            eval_index_list_part.sort()
            if eval_index_list is None:
                eval_index_list = np.array(
                    [[eval_index, i] for eval_index in eval_index_list_part]
                )
            else:
                eval_index_list = np.append(
                    eval_index_list,
                    [[eval_index, i] for eval_index in eval_index_list_part],
                    axis=0,
                )
            print(eval_index_list.shape)
        eval_index_list_eval_worker = np.split(
            eval_index_list, conf.number_eval_workers
        )
    else:
        if conf.train_case_study_mode == "crash_nearmiss":
            tmp_safe_id_list = list(
                set(conf.data_info_origin["safe_ep_info"])
                - set(data_info_new["crashnearmiss_history"]["safe2crash_ep_info"])
            )
            tmp_safe_weight_arr = np.array(
                [ind_info[-1] for ind_info in tmp_safe_id_list]
            )
            tmp_safe_id_list = [ind_info[0:-1] for ind_info in tmp_safe_id_list]

            tmp_crash_id_list = [
                list(ele) for ele in conf.data_info_origin["crash_id_list"]
            ]
            tmp_crash_weight = list(conf.data_info_origin["crash_weight"])

            # split safe traj into weight>=1 and weight<1 pool
            safe_traj_num = (
                conf.number_eval_workers * conf.ep_num_eval_worker
                - len(data_info_new["crashnearmiss_history"]["safe2crash_ep_info"])
                - len(conf.data_info_origin["crash_id_list"])
            )
            assert (
                safe_traj_num >= 0
            ), f"Total number of evaluated episodes is {conf.number_eval_workers}x{conf.ep_num_eval_worker} is smaller than {len(data_info_new['crashnearmiss_history']['safe2crash_ep_info'])}+{len(conf.data_info_origin['crash_id_list'])}, please increase the variables number_eval_workers or ep_num_eval_worker"
            safe_traj_weight1_index_list = np.where(tmp_safe_weight_arr > 0.99)
            safe_traj_weight0_index_list = np.where(tmp_safe_weight_arr <= 0.99)
            safe_traj_weight1_sample_num = min(
                int(safe_traj_num * conf.data_info_weight1_ratio),
                len(safe_traj_weight1_index_list[0]),
            )  # ensure safe_traj_weight1_sample_num is not greater than the size of safe_traj_weight1_index_list
            safe_traj_weight0_sample_num = safe_traj_num - safe_traj_weight1_sample_num
            safe_traj_weight1_weight_list = tmp_safe_weight_arr[
                safe_traj_weight1_index_list
            ]
            safe_traj_weight0_weight_list = tmp_safe_weight_arr[
                safe_traj_weight0_index_list
            ]
            p_weight1 = safe_traj_weight1_weight_list / np.sum(
                safe_traj_weight1_weight_list
            )
            p_weight0 = safe_traj_weight0_weight_list / np.sum(
                safe_traj_weight0_weight_list
            )
            eval_index_list1 = np.random.choice(
                safe_traj_weight1_index_list[0],
                safe_traj_weight1_sample_num,
                replace=False,
                p=p_weight1,
            )
            assert safe_traj_weight0_sample_num <= len(
                safe_traj_weight0_weight_list
            ), f"{safe_traj_weight0_sample_num} larger than {len(safe_traj_weight0_weight_list)}, please reduce the variables number_eval_workers or ep_num_eval_worker"
            eval_index_list0 = np.random.choice(
                safe_traj_weight0_index_list[0],
                safe_traj_weight0_sample_num,
                replace=False,
                p=p_weight0,
            )
            print(
                "new custom_eval_function",
                "weight>=1",
                safe_traj_weight1_sample_num,
                len(safe_traj_weight1_weight_list),
            )
            print(
                "new custom_eval_function",
                "weight<1",
                safe_traj_weight0_sample_num,
                len(safe_traj_weight0_weight_list),
            )
            eval_index_list = np.append(eval_index_list0, eval_index_list1)

            eval_index_list = np.append(
                eval_index_list,
                np.arange(
                    len(tmp_safe_id_list),
                    len(tmp_safe_id_list)
                    + len(data_info_new["crashnearmiss_history"]["safe2crash_ep_info"])
                    + len(conf.data_info_origin["crash_id_list"]),
                ),
            )
            tmp_safe_id_list += [
                ind_info[0:-1]
                for ind_info in data_info_new["crashnearmiss_history"][
                    "safe2crash_ep_info"
                ]
            ]
            tmp_safe_weight_arr = np.append(
                tmp_safe_weight_arr,
                np.array(
                    [
                        ind_info[-1]
                        for ind_info in data_info_new["crashnearmiss_history"][
                            "safe2crash_ep_info"
                        ]
                    ]
                ),
            )
            print(
                "start custom_eval_function",
                len(data_info_new["crash_id_list"]),
                len(data_info_new["safe2crash_id_list"]),
                len(tmp_safe_id_list),
            )
            eval_index_list.sort()
            eval_index_list_eval_worker = np.split(
                eval_index_list, conf.number_eval_workers
            )
        elif conf.train_case_study_mode == "crash":
            # only consider crash trajectories
            tmp_safe_id_list = []
            tmp_safe_weight_arr = np.array([])

            tmp_crash_id_list = [
                list(ele) for ele in conf.data_info_origin["crash_id_list"]
            ]
            tmp_crash_weight = list(conf.data_info_origin["crash_weight"])

            eval_index_list = np.arange(len(conf.data_info_origin["crash_id_list"]))

            print("start custom_eval_function", len(data_info_new["crash_id_list"]))
            eval_index_list.sort()
            eval_index_list_eval_worker = np.split(
                eval_index_list, conf.number_eval_workers
            )
    return (
        tmp_safe_id_list,
        tmp_safe_weight_arr,
        tmp_crash_id_list,
        tmp_crash_weight,
        eval_index_list_eval_worker,
    )


def custom_eval_function2(trainer, eval_workers):
    """Example of a custom evaluation function.
    Args:
        trainer (Trainer): Trainer instance to evaluate the latest policy.
        eval_workers (WorkerSet): Evaluation workers.
    Returns:
        metrics (Dict): Results of the evaluation metrics.
    """
    t0 = time.time()
    load_data_info_origin()
    data_info_new = load_data_info_new()
    t01 = time.time()

    (
        tmp_safe_id_list,
        tmp_safe_weight_arr,
        tmp_crash_id_list,
        tmp_crash_weight,
        eval_index_list_eval_worker,
    ) = sample_eval_index_list_eval_worker(data_info_new)
    t1 = time.time()

    worker_index = 0
    for worker in eval_workers.remote_workers():
        worker.foreach_env.remote(lambda env: env.set_eval_index())
        worker.foreach_env.remote(
            lambda env: env.set_traj_pool(
                tmp_safe_id_list,
                tmp_safe_weight_arr,
                tmp_crash_id_list,
                tmp_crash_weight,
            )
        )
        worker.foreach_env.remote(
            lambda env: env.set_eval_trajs(eval_index_list_eval_worker)
        )
        worker_index += 1
    t3 = time.time()

    for i in range(conf.eval_round):
        # print("Custom evaluation round", i)
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])
    t2 = time.time()

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999
    )

    if isinstance(conf.data_info_origin, list):
        multiple_model = True
    else:
        multiple_model = False

    interest_time = np.array([0.0, 0.0, 0.0, 0.0])
    interest_time2 = 0.0
    safe2crash_info, safe2nearmiss_info = [], []
    if multiple_model:
        original_crash_stats = [0] * (len(conf.data_info_origin) * 3)
    else:
        original_crash_stats = [0, 0, 0]
    eval_ep_id_list = []

    for ep in episodes:
        if multiple_model:
            folder_index = ep.hist_data["data_folder_index"][0]
            eval_ep_id_list.append([ep.hist_data["ep_id"][0], folder_index])
            # crash
            if len(ep.hist_data["safe2crash_id"][0]) > 0:
                # new crash: add to crash/near-miss pool, safe score = 0
                if (
                    ep.hist_data["safe2crash_id"][0]
                    not in data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_id_list"
                    ]
                ):
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_id_list"
                    ].append(ep.hist_data["safe2crash_id"][0])
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_weight"
                    ].append(ep.hist_data["safe2crash_weight"][0])
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_ep_info"
                    ].append(
                        tuple(
                            ep.hist_data["safe2crash_id"][0]
                            + [ep.hist_data["safe2crash_weight"][0]]
                        )
                    )
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_score_list"
                    ].append(
                        [1, 0, 0, 0]
                    )  # crash, near-miss, non-crash, crash/near-miss/safe index
                    safe2crash_info.append(ep.hist_data["safe2crash_id"][0])
                # previous crash/near-miss
                else:
                    index = data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_id_list"
                    ].index(ep.hist_data["safe2crash_id"][0])
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_score_list"
                    ][index][0] += 1
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_score_list"
                    ][index][3] = 0
            # near miss
            elif len(ep.hist_data["safe2nearmiss_id"][0]) > 0:
                # new near-miss: add to crash/near-miss pool, update safe score
                if (
                    ep.hist_data["safe2nearmiss_id"][0]
                    not in data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_id_list"
                    ]
                ):
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_id_list"
                    ].append(ep.hist_data["safe2nearmiss_id"][0])
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_weight"
                    ].append(ep.hist_data["safe2nearmiss_weight"][0])
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_ep_info"
                    ].append(
                        tuple(
                            ep.hist_data["safe2nearmiss_id"][0]
                            + [ep.hist_data["safe2nearmiss_weight"][0]]
                        )
                    )
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_score_list"
                    ].append([0, 1, 0, 1])
                    safe2nearmiss_info.append(ep.hist_data["safe2nearmiss_id"][0])
                # previous crash/near-miss
                else:
                    index = data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_id_list"
                    ].index(ep.hist_data["safe2nearmiss_id"][0])
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_score_list"
                    ][index][1] += 1
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_score_list"
                    ][index][3] = 1
            # safety-critical non-crash
            elif len(ep.hist_data["safe2safe_id"][0]) > 0:
                # previous crash/near-miss: update safe score
                if (
                    ep.hist_data["safe2safe_id"][0]
                    in data_info_new[folder_index]["safe2crash_id_list"]
                ):
                    index = data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_id_list"
                    ].index(ep.hist_data["safe2safe_id"][0])
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_score_list"
                    ][index][2] += 1
                    data_info_new[folder_index]["crashnearmiss_history"][
                        "safe2crash_score_list"
                    ][index][3] = 2
            # crash
            elif len(ep.hist_data["crash2crash_id"][0]) > 0:
                # previous crash
                assert (
                    ep.hist_data["crash2crash_id"][0]
                    in data_info_new[folder_index]["crashnearmiss_history"][
                        "crash_id_list"
                    ]
                )
                index = data_info_new[folder_index]["crashnearmiss_history"][
                    "crash_id_list"
                ].index(ep.hist_data["crash2crash_id"][0])
                data_info_new[folder_index]["crashnearmiss_history"][
                    "crash_score_list"
                ][index][0] += 1
                data_info_new[folder_index]["crashnearmiss_history"][
                    "crash_score_list"
                ][index][3] = 0
                original_crash_stats[folder_index * 3] += data_info_new[folder_index][
                    "crashnearmiss_history"
                ]["crash_weight"][index]
            # near-miss
            elif len(ep.hist_data["crash2nearmiss_id"][0]) > 0:
                # previous crash
                assert (
                    ep.hist_data["crash2nearmiss_id"][0]
                    in data_info_new[folder_index]["crashnearmiss_history"][
                        "crash_id_list"
                    ]
                )
                index = data_info_new[folder_index]["crashnearmiss_history"][
                    "crash_id_list"
                ].index(ep.hist_data["crash2nearmiss_id"][0])
                data_info_new[folder_index]["crashnearmiss_history"][
                    "crash_score_list"
                ][index][1] += 1
                data_info_new[folder_index]["crashnearmiss_history"][
                    "crash_score_list"
                ][index][3] = 1
                original_crash_stats[folder_index * 3 + 1] += data_info_new[
                    folder_index
                ]["crashnearmiss_history"]["crash_weight"][index]
            # safety-critical non-crash
            elif len(ep.hist_data["crash2safe_id"][0]) > 0:
                # previous crash
                assert (
                    ep.hist_data["crash2safe_id"][0]
                    in data_info_new[folder_index]["crashnearmiss_history"][
                        "crash_id_list"
                    ]
                )
                index = data_info_new[folder_index]["crashnearmiss_history"][
                    "crash_id_list"
                ].index(ep.hist_data["crash2safe_id"][0])
                data_info_new[folder_index]["crashnearmiss_history"][
                    "crash_score_list"
                ][index][2] += 1
                data_info_new[folder_index]["crashnearmiss_history"][
                    "crash_score_list"
                ][index][3] = 2
                original_crash_stats[folder_index * 3 + 2] += data_info_new[
                    folder_index
                ]["crashnearmiss_history"]["crash_weight"][index]
        else:
            eval_ep_id_list.append(ep.hist_data["ep_id"][0])
            # crash
            if len(ep.hist_data["safe2crash_id"][0]) > 0:
                # new crash: add to crash/near-miss pool, safe score = 0
                if (
                    ep.hist_data["safe2crash_id"][0]
                    not in data_info_new["crashnearmiss_history"]["safe2crash_id_list"]
                ):
                    data_info_new["crashnearmiss_history"]["safe2crash_id_list"].append(
                        ep.hist_data["safe2crash_id"][0]
                    )
                    data_info_new["crashnearmiss_history"]["safe2crash_weight"].append(
                        ep.hist_data["safe2crash_weight"][0]
                    )
                    data_info_new["crashnearmiss_history"]["safe2crash_ep_info"].append(
                        tuple(
                            ep.hist_data["safe2crash_id"][0]
                            + [ep.hist_data["safe2crash_weight"][0]]
                        )
                    )
                    data_info_new["crashnearmiss_history"][
                        "safe2crash_score_list"
                    ].append(
                        [1, 0, 0, 0]
                    )  # crash, near-miss, non-crash, crash/near-miss/safe index
                    safe2crash_info.append(ep.hist_data["safe2crash_id"][0])
                # previous crash/near-miss
                else:
                    index = data_info_new["crashnearmiss_history"][
                        "safe2crash_id_list"
                    ].index(ep.hist_data["safe2crash_id"][0])
                    data_info_new["crashnearmiss_history"]["safe2crash_score_list"][
                        index
                    ][0] += 1
                    data_info_new["crashnearmiss_history"]["safe2crash_score_list"][
                        index
                    ][3] = 0
            # near miss
            elif len(ep.hist_data["safe2nearmiss_id"][0]) > 0:
                # new near-miss: add to crash/near-miss pool, update safe score
                if (
                    ep.hist_data["safe2nearmiss_id"][0]
                    not in data_info_new["crashnearmiss_history"]["safe2crash_id_list"]
                ):
                    data_info_new["crashnearmiss_history"]["safe2crash_id_list"].append(
                        ep.hist_data["safe2nearmiss_id"][0]
                    )
                    data_info_new["crashnearmiss_history"]["safe2crash_weight"].append(
                        ep.hist_data["safe2nearmiss_weight"][0]
                    )
                    data_info_new["crashnearmiss_history"]["safe2crash_ep_info"].append(
                        tuple(
                            ep.hist_data["safe2nearmiss_id"][0]
                            + [ep.hist_data["safe2nearmiss_weight"][0]]
                        )
                    )
                    data_info_new["crashnearmiss_history"][
                        "safe2crash_score_list"
                    ].append([0, 1, 0, 1])
                    safe2nearmiss_info.append(ep.hist_data["safe2nearmiss_id"][0])
                # previous crash/near-miss
                else:
                    index = data_info_new["crashnearmiss_history"][
                        "safe2crash_id_list"
                    ].index(ep.hist_data["safe2nearmiss_id"][0])
                    data_info_new["crashnearmiss_history"]["safe2crash_score_list"][
                        index
                    ][1] += 1
                    data_info_new["crashnearmiss_history"]["safe2crash_score_list"][
                        index
                    ][3] = 1
            # safety-critical non-crash
            elif len(ep.hist_data["safe2safe_id"][0]) > 0:
                # previous crash/near-miss: update safe score
                if (
                    ep.hist_data["safe2safe_id"][0]
                    in data_info_new["safe2crash_id_list"]
                ):
                    index = data_info_new["crashnearmiss_history"][
                        "safe2crash_id_list"
                    ].index(ep.hist_data["safe2safe_id"][0])
                    data_info_new["crashnearmiss_history"]["safe2crash_score_list"][
                        index
                    ][2] += 1
                    data_info_new["crashnearmiss_history"]["safe2crash_score_list"][
                        index
                    ][3] = 2
            # crash
            elif len(ep.hist_data["crash2crash_id"][0]) > 0:
                # previous crash
                assert (
                    ep.hist_data["crash2crash_id"][0]
                    in data_info_new["crashnearmiss_history"]["crash_id_list"]
                )
                index = data_info_new["crashnearmiss_history"]["crash_id_list"].index(
                    ep.hist_data["crash2crash_id"][0]
                )
                data_info_new["crashnearmiss_history"]["crash_score_list"][index][
                    0
                ] += 1
                data_info_new["crashnearmiss_history"]["crash_score_list"][index][3] = 0
                original_crash_stats[0] += data_info_new["crashnearmiss_history"][
                    "crash_weight"
                ][index]
            # near-miss
            elif len(ep.hist_data["crash2nearmiss_id"][0]) > 0:
                # previous crash
                assert (
                    ep.hist_data["crash2nearmiss_id"][0]
                    in data_info_new["crashnearmiss_history"]["crash_id_list"]
                )
                index = data_info_new["crashnearmiss_history"]["crash_id_list"].index(
                    ep.hist_data["crash2nearmiss_id"][0]
                )
                data_info_new["crashnearmiss_history"]["crash_score_list"][index][
                    1
                ] += 1
                data_info_new["crashnearmiss_history"]["crash_score_list"][index][3] = 1
                original_crash_stats[1] += data_info_new["crashnearmiss_history"][
                    "crash_weight"
                ][index]
            # safety-critical non-crash
            elif len(ep.hist_data["crash2safe_id"][0]) > 0:
                # previous crash
                assert (
                    ep.hist_data["crash2safe_id"][0]
                    in data_info_new["crashnearmiss_history"]["crash_id_list"]
                )
                index = data_info_new["crashnearmiss_history"]["crash_id_list"].index(
                    ep.hist_data["crash2safe_id"][0]
                )
                data_info_new["crashnearmiss_history"]["crash_score_list"][index][
                    2
                ] += 1
                data_info_new["crashnearmiss_history"]["crash_score_list"][index][3] = 2
                original_crash_stats[2] += data_info_new["crashnearmiss_history"][
                    "crash_weight"
                ][index]
        for time_list in ep.hist_data["interest_time"]:
            if time_list is not None:
                interest_time = interest_time + np.array(time_list)
        interest_time2 += sum(ep.hist_data["interest_time2"])

    print("eval num:", len(eval_ep_id_list), "; samples:", eval_ep_id_list[0:10])

    if isinstance(data_info_new, list):
        data_info_new = update_based_on_safe_score_new_multiple(data_info_new)
        for i in range(len(conf.data_info_new_path)):
            with open(conf.data_info_new_path[i], "w") as fp:
                ujson.dump(data_info_new[i], fp)
    else:
        data_info_new = update_based_on_safe_score_new(data_info_new)
        with open(conf.data_info_new_path, "w") as fp:
            ujson.dump(data_info_new, fp)
        iter_data_store_path = conf.data_info_new_path.replace(
            ".json", f"_{trainer.iteration}.json"
        )
        with open(iter_data_store_path, "w") as fp:
            ujson.dump(data_info_new, fp)

    metrics = {
        "safe2crash_info": safe2crash_info,
        "safe2nearmiss_info": safe2nearmiss_info,
        "eval_worker_number": worker_index,
        "eval_episode_number": len(episodes),
        "eval_time": [
            t1 - t0,
            t01 - t0,
            t3 - t1,
            t2 - t3,
            time.time() - t2,
            interest_time,
            interest_time2,
        ],
        "original_crash_stats": list(original_crash_stats),
        "eval_ep_id_list": list(eval_ep_id_list),
    }
    return metrics


def update_based_on_safe_score_new(data_info_dict):
    """Update the information of the latest training dataset based on the evaluation results.

    Args:
        data_info_dict (Dict): Information of the latest training dataset.

    Returns:
        Dict: Updated information of the training dataset.
    """
    new_data_info_dict = {
        "crash_id_list": [],
        "crash_weight": [],
        "crash_ep_info": [],
        "safe2crash_id_list": [],
        "safe2crash_weight": [],
        "safe2crash_ep_info": [],
        "crashnearmiss_history": data_info_dict["crashnearmiss_history"],
    }
    for i in range(len(data_info_dict["crashnearmiss_history"]["crash_score_list"])):
        crash_score_info = data_info_dict["crashnearmiss_history"]["crash_score_list"][
            i
        ]
        remove_flag = False
        if (
            sum(crash_score_info[0:-1]) >= evaluate_times_threshold
            and crash_score_info[3] == 2
        ):  # enough evaluation iterations and currently safe
            remove_flag = True

        if not remove_flag:
            new_data_info_dict["crash_id_list"].append(
                data_info_dict["crashnearmiss_history"]["crash_id_list"][i]
            )
            new_data_info_dict["crash_weight"].append(
                data_info_dict["crashnearmiss_history"]["crash_weight"][i]
            )
            new_data_info_dict["crash_ep_info"].append(
                data_info_dict["crashnearmiss_history"]["crash_ep_info"][i]
            )

    for i in range(
        len(data_info_dict["crashnearmiss_history"]["safe2crash_score_list"])
    ):
        safe2crash_score_info = data_info_dict["crashnearmiss_history"][
            "safe2crash_score_list"
        ][i]
        remove_flag = False
        if (
            sum(safe2crash_score_info[0:-1]) >= evaluate_times_threshold
            and safe2crash_score_info[3] == 2
        ):  # enough evaluation iterations and currently safe
            remove_flag = True

        if not remove_flag:
            new_data_info_dict["safe2crash_id_list"].append(
                data_info_dict["crashnearmiss_history"]["safe2crash_id_list"][i]
            )
            new_data_info_dict["safe2crash_weight"].append(
                data_info_dict["crashnearmiss_history"]["safe2crash_weight"][i]
            )
            new_data_info_dict["safe2crash_ep_info"].append(
                data_info_dict["crashnearmiss_history"]["safe2crash_ep_info"][i]
            )
    print(
        "end custom_eval_function",
        len(new_data_info_dict["crash_id_list"]),
        len(new_data_info_dict["safe2crash_id_list"]),
    )
    return new_data_info_dict


def update_based_on_safe_score_new_multiple(data_info_dict):
    """Update multiple training datasets based on the evaluation results.

    Args:
        data_info_dict (list[Dict]): List of the information for multiple training datasets.

    Returns:
        list[Dict]: List of updated information for multiple training datasets.
    """
    new_data_info_dict = []
    for j in range(len(data_info_dict)):
        new_data_info_dict_part = {
            "crash_id_list": [],
            "crash_weight": [],
            "crash_ep_info": [],
            "safe2crash_id_list": [],
            "safe2crash_weight": [],
            "safe2crash_ep_info": [],
            "crashnearmiss_history": data_info_dict[j]["crashnearmiss_history"],
        }
        for i in range(
            len(data_info_dict[j]["crashnearmiss_history"]["crash_score_list"])
        ):
            crash_score_info = data_info_dict[j]["crashnearmiss_history"][
                "crash_score_list"
            ][i]
            remove_flag = False
            if (
                sum(crash_score_info[0:-1]) >= evaluate_times_threshold
                and crash_score_info[3] == 2
            ):  # enough evaluation iterations and currently safe
                remove_flag = True

            if not remove_flag:
                new_data_info_dict_part["crash_id_list"].append(
                    data_info_dict[j]["crashnearmiss_history"]["crash_id_list"][i]
                )
                new_data_info_dict_part["crash_weight"].append(
                    data_info_dict[j]["crashnearmiss_history"]["crash_weight"][i]
                )
                new_data_info_dict_part["crash_ep_info"].append(
                    data_info_dict[j]["crashnearmiss_history"]["crash_ep_info"][i]
                )

        for i in range(
            len(data_info_dict[j]["crashnearmiss_history"]["safe2crash_score_list"])
        ):
            safe2crash_score_info = data_info_dict[j]["crashnearmiss_history"][
                "safe2crash_score_list"
            ][i]
            remove_flag = False
            if (
                sum(safe2crash_score_info[0:-1]) >= evaluate_times_threshold
                and safe2crash_score_info[3] == 2
            ):  # enough evaluation iterations and currently safe
                remove_flag = True

            if not remove_flag:
                new_data_info_dict_part["safe2crash_id_list"].append(
                    data_info_dict[j]["crashnearmiss_history"]["safe2crash_id_list"][i]
                )
                new_data_info_dict_part["safe2crash_weight"].append(
                    data_info_dict[j]["crashnearmiss_history"]["safe2crash_weight"][i]
                )
                new_data_info_dict_part["safe2crash_ep_info"].append(
                    data_info_dict[j]["crashnearmiss_history"]["safe2crash_ep_info"][i]
                )
        print(
            "end custom_eval_function",
            len(new_data_info_dict_part["crash_id_list"]),
            len(new_data_info_dict_part["safe2crash_id_list"]),
        )
        new_data_info_dict.append(new_data_info_dict_part)
    return new_data_info_dict


print("Nodes in the Ray cluster:")
print(ray.nodes())
tune.run(
    "PPO",
    stop={"training_iteration": 2000},
    config={
        "env": "my_env",
        "num_gpus": 0,
        "num_workers": conf.experiment_config["num_workers"],
        "num_envs_per_worker": 1,
        "lr": conf.experiment_config["lr"],
        "gamma": 0.99,  # 1
        "train_batch_size": conf.experiment_config["train_batch_size"],
        "batch_mode": "complete_episodes",
        "framework": "torch",
        "ignore_worker_failures": True,
        "callbacks": MyCallbacks,
        "evaluation_config": {
            # Example: overriding env_config, exploration, etc:
            "explore": False
        },
        "evaluation_num_workers": conf.number_eval_workers,
        # Optional custom eval function.
        "custom_eval_function": custom_eval_function2,
        # Enable evaluation, once per training iteration.
        "evaluation_interval": 1,
    },
    checkpoint_freq=1,
    restore=conf.experiment_config["restore_path"],
    local_dir=conf.experiment_config["root_folder"],
    name=conf.experiment_config["experiment_name"],
)
