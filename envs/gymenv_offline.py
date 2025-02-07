from gym import core
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import os, glob
import json, ujson
import time

from mtlsp.simulator import Simulator
from mtlsp.utils import json2dict
from .nde import NDE
from .nade import NADE
import os
if os.environ["mode"] == "testing":
    import conf.conf_testing as conf
elif os.environ["mode"] == "training":
    import conf.conf_training as conf
else:
    raise ValueError("Please set the mode to testing or training")
from controller.rlcontroller import RLControllerNew
from controller.treesearchnadecontroller import TreeSearchNADEBackgroundController


class RL_NDE_offline(core.Env):
    def __init__(self, env_config=None, train_flag=True):
        """Initialize the RL_NDE_offline class.

        Args:
            env_config (dict): Configuration of the environment.
            train_flag (bool): Whether the GYM environment is in training mode.
        """
        self.env = None
        self.simulator = None
        # Actions should include 31 accelerations (-4:0.2:2 m/s^2) and 10 steering angles (-0.5:0.1:-0.1, 0.1:0.1:0.5 deg) 
        self.action_space = Box(low=np.float32(np.array([-4,-10])), high=np.float32(np.array([2,10])), shape=(2, ))
        self.observation_space = Box(low=np.float32(np.ones(43)*(-1)), high=np.float32(np.ones(43)), shape=(43, ))
        self.cav_speed_list = []
        self.cav_mean_speed = 0
        self.obs_and_action = []
        self.reward_normal, self.reward_time, self.reward_crash, self.reward_leave, self.reward_spendrange, self.reward_lanekeep = 0,0,0,0,0,0
        self.train_flag = train_flag
        self.critical_flag = False
        self.sub_critical_time_step_num = 0
        self.critical_time_step_num = 0
        self.critical_time_rss_num = 0
        self.highly_critical = False
        self.slightly_critical = False
        self.total_episode = 0
        self.total_step = 0
        # load all data of crash and safe
        if conf.data_info_origin is not None:
            self.data_info_origin = conf.data_info_origin
        else:
            if isinstance(conf.data_info_origin_path, list) and len(conf.data_info_origin_path) > 1:
                self.data_info_origin = []
                for p_ in conf.data_info_origin_path:
                    with open(p_) as file_obj:
                        data_info_origin_part = ujson.load(file_obj)
                    data_info_origin_part["safe_ep_info"] = [tuple(info) for info in data_info_origin_part["safe_ep_info"]]
                    self.data_info_origin.append(data_info_origin_part)
            else: # assume data_info_origin_path is a string
                with open(conf.data_info_origin_path) as file_obj:
                    self.data_info_origin = ujson.load(file_obj)
                self.data_info_origin["safe_ep_info"] = [tuple(info) for info in self.data_info_origin["safe_ep_info"]]
        
        if conf.train_case_study_mode != "crash":
            # load near-miss data
            if isinstance(conf.data_info_new_path, list) and len(conf.data_info_new_path) > 1:
                data_info_new = []
                for p_ in conf.data_info_new_path:
                    with open(p_) as file_obj:
                        data_info_new_part = ujson.load(file_obj)
                    data_info_new_part["safe2crash_ep_info"] = [tuple(info) for info in data_info_new_part["safe2crash_ep_info"]]
                    data_info_new.append(data_info_new_part)
            else: # asssume data_info_new_path is a string
                with open(conf.data_info_new_path) as file_obj2:
                    data_info_new = ujson.load(file_obj2)
                data_info_new["safe2crash_ep_info"] = [tuple(info) for info in data_info_new["safe2crash_ep_info"]]
            if isinstance(self.data_info_origin, list) and isinstance(data_info_new, list):
                tmp_safe_ep_info,tmp_safe_weight_arr,tmp_safe_id_list = [],[],[]
                for i in range(len(self.data_info_origin)):
                    tmp_safe_ep_info_part = list(set(self.data_info_origin[i]["safe_ep_info"])-set(data_info_new[i]["safe2crash_ep_info"]))
                    tmp_safe_weight_arr_part = np.array([ind_info[-1] for ind_info in tmp_safe_ep_info_part])
                    tmp_safe_id_list_part = [ind_info[0:-1] for ind_info in tmp_safe_ep_info_part]
                    tmp_safe_ep_info.append(tmp_safe_ep_info_part)
                    tmp_safe_weight_arr.append(tmp_safe_weight_arr_part)
                    tmp_safe_id_list.append(tmp_safe_id_list_part)
            else:
                tmp_safe_ep_info = list(set(self.data_info_origin["safe_ep_info"])-set(data_info_new["safe2crash_ep_info"]))
                tmp_safe_weight_arr = np.array([ind_info[-1] for ind_info in tmp_safe_ep_info])
                tmp_safe_id_list = [ind_info[0:-1] for ind_info in tmp_safe_ep_info]
            # load crash
            if not conf.update_crash:
                crash_source = self.data_info_origin
            else:
                crash_source = data_info_new
            if isinstance(crash_source, list):
                tmp_crash_id_list,tmp_crash_weight,num_crash = [],[],[]
                for i in range(len(crash_source)):
                    tmp_crash_id_list_part = [list(ele) for ele in crash_source[i]["crash_id_list"]]
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
            self.set_traj_pool(tmp_safe_id_list, tmp_safe_weight_arr, tmp_crash_id_list, tmp_crash_weight, num_crash)
        else:
            self.set_traj_pool([], [], [list(ele) for ele in self.data_info_origin["crash_id_list"]], list(self.data_info_origin["crash_weight"]), len(self.data_info_origin["crash_id_list"]))
        # assert(self.data_info_origin is not None)
        self.data_info_new = None
        self.safe2crash_ep_id = []
        self.safe2crash_ep_weight = None
        self.safe2nearmiss_ep_id = []
        self.safe2nearmiss_ep_weight = None
        self.safe2safe_ep_id = []
        self.safe2safe_ep_weight = None
        self.crash2crash_ep_id = []
        self.crash2crash_ep_weight = None
        self.crash2nearmiss_ep_id = []
        self.crash2nearmiss_ep_weight = None
        self.crash2safe_ep_id = []
        self.crash2safe_ep_weight = None
        self.data_folder_index = None
        self.sampled_crash = self.sampled_crash_still_crash = 0
        self.eval_flag = False
        self.worker_index = None
        self.eval_index_list = None
        self.init_wall_time = time.time()
        self.loaded_safe_files_index = None
        self.fcd_trajs = {}
        self.json_trajs = {}
        self.env_config = env_config
        if self.env_config is not None:
            self.worker_index = f"{self.env_config.worker_index}/{self.env_config.num_workers}"
        self.ep_id = None

    def select_replay_file_train(self):
        """Select the replay file for collecting training batch.

        Returns:
            list: Information of the replay file.
        """
        # ratio = 1/2.
        ratio  = 1.
        
        if random.random() >= ratio:
            # safe trajectory
            index = random.choices(range(len(self.tmp_safe_id_list)),weights=self.tmp_safe_weight)[0]
            input_info = [
                list(self.tmp_safe_id_list[index]),
                os.path.join(conf.data_folder[0], "tested_and_safe"),
                self.tmp_safe_weight[index],
                0,
                "RLNew",
                0
            ]
            self.sampled_crash = 0
            self.data_folder_index = 0
        else:
            # crash trajectory
            if conf.data_folder_additional_info is not None:
                folder_index = random.choices(range(len(conf.data_folder)),weights=conf.data_folder_additional_info["NDE_crash_rate"])[0]
                index = random.choices(range(len(self.tmp_crash_id_list[folder_index])),weights=self.tmp_crash_weight[folder_index])[0]
                if index >= self.num_crash[folder_index]:
                    folder_name = "tested_and_safe"
                    self.sampled_crash = 0
                else:
                    folder_name = "crash"
                    self.sampled_crash = 1
                input_info = [
                    list(self.tmp_crash_id_list[folder_index][index]),
                    os.path.join(conf.data_folder[folder_index], folder_name),
                    self.tmp_crash_weight[folder_index][index],
                    1,
                    conf.data_folder_additional_info["cav_model"][folder_index],
                    folder_index
                ]
            else:
                folder_index = 0
                index = random.choices(range(len(self.tmp_crash_id_list)),weights=self.tmp_crash_weight)[0]
                if index >= self.num_crash:
                    folder_name = "tested_and_safe"
                    self.sampled_crash = 0
                else:
                    folder_name = "crash"
                    self.sampled_crash = 1
                input_info = [
                    list(self.tmp_crash_id_list[index]),
                    os.path.join(conf.data_folder[0], folder_name),
                    self.tmp_crash_weight[index],
                    1,
                    "RLNew",
                    0
                ]
            self.data_folder_index = folder_index
        # print(input_info)
        return input_info

    def select_replay_file_eval(self):
        """Select the replay file for collecting evaluation batch.

        Returns:
            list: Information of the replay file.
        """
        # num_ep = len(self.eval_trajs)
        # index = self.eval_index % num_ep
        # input_info = [list(self.eval_trajs[index][2])]
        # input_info.append(self.eval_trajs[index][0:2])
        # input_info.append(input_info[1][1]["weight_episode"])
        # input_info.append(0)
        num_ep = len(self.eval_index_list)
        index = self.eval_index_list[self.eval_index % num_ep]
        # print("select_replay_file_eval", self.eval_index, index)
        if isinstance(index, np.ndarray):
            traj_index = index[0]
            folder_index = index[1]
            if traj_index < len(self.tmp_safe_id_list[folder_index]):
                input_info = [
                    list(self.tmp_safe_id_list[folder_index][traj_index]),
                    os.path.join(conf.data_folder[folder_index], "tested_and_safe"),
                    self.tmp_safe_weight[folder_index][traj_index],
                    0,
                    conf.data_folder_additional_info["cav_model"][folder_index],
                    folder_index
                ]
            else:
                input_info = [
                    list(self.tmp_crash_id_list[folder_index][traj_index-len(self.tmp_safe_id_list[folder_index])]),
                    os.path.join(conf.data_folder[folder_index], "crash"),
                    self.tmp_crash_weight[folder_index][traj_index-len(self.tmp_safe_id_list[folder_index])],
                    1,
                    conf.data_folder_additional_info["cav_model"][folder_index],
                    folder_index
                ]
        else:
            folder_index = 0
            if index < len(self.tmp_safe_id_list):
                input_info = [
                    list(self.tmp_safe_id_list[index]),
                    os.path.join(conf.data_folder[0], "tested_and_safe"),
                    self.tmp_safe_weight[index],
                    0,
                    "RLNew",
                    0
                ]
            else:
                input_info = [
                    list(self.tmp_crash_id_list[index-len(self.tmp_safe_id_list)]),
                    os.path.join(conf.data_folder[0], "crash"),
                    self.tmp_crash_weight[index-len(self.tmp_safe_id_list)],
                    1,
                    "RLNew",
                    0
                ]
        self.data_folder_index = folder_index
        self.eval_index += 1
        # print(input_info)
        return input_info

    def _reset(self):
        """Helper function to reset the environment.

        Returns:
            np.array: Initial observation of the environment.
        """
        self.total_episode += 1
        # reset environment
        if self.simulator is not None:
            self.simulator.stop()
        sumo_config_file_path = "/maps/3LaneHighway/3LaneHighway.sumocfg"
        if not self.eval_flag:
            input_file = self.select_replay_file_train()
        else:
            input_file = self.select_replay_file_eval()      
        if conf.env_mode == "NDE":
            self.env = NDE(AVController=RLControllerNew)
        elif conf.env_mode == "NADE":
            self.env = NADE(BVController=TreeSearchNADEBackgroundController, cav_model=input_file[4])
        self.env.eval_flag = self.eval_flag
        self.ep_id = int(input_file[0][0])
        self.simulator = Simulator(
			sumo_net_file_path=conf.experiment_config["code_root_folder"] + '/maps/3LaneHighway/3LaneHighway.net.xml',
			sumo_config_file_path=conf.experiment_config["code_root_folder"] + sumo_config_file_path,
			num_tries=50,
			step_size=0.1,
			action_step_size=0.1,
			lc_duration=1,
			track_cav=False,
			sublane_flag=True,
			gui_flag=False,
            input=input_file,
			output=None,
			experiment_path=None
		)
        obs = np.ones(43)
        if self.train_flag:
            self.simulator.bind_env(self.env)
            self.simulator.start()
            self.env.initialize()
            self.simulator.plain_traci_step()
            self.env._check_vehicle_list()
            obs = self._get_observation()
            if conf.precise_criticality_flag:
                # obs, reward, done, info = self.fast_step()
                done, reason = self._get_done()
                if done:
                    return self._reset()
        return obs

    def reset(self):
        """Main function to reset the environment.

        Returns:
            np.array: Initial observation of the environment.
        """
        # reset parameters
        self.init_wall_time = time.time()
        self.cav_speed_list = []
        self.cav_mean_speed = 0
        self.obs_and_action = []
        self.reward_normal, self.reward_time, self.reward_crash, self.reward_leave, self.reward_spendrange, self.reward_lanekeep = 0,0,0,0,0,0
        self.critical_flag = False
        self.sub_critical_time_step_num = 0
        self.critical_time_step_num = 0
        self.critical_time_rss_num = 0
        self.highly_critical = False
        self.slightly_critical = False
        self.total_episode = 0
        self.total_step = 0
        self.safe2crash_ep_id = []
        self.safe2crash_ep_weight = None
        self.safe2nearmiss_ep_id = []
        self.safe2nearmiss_ep_weight = None
        self.safe2safe_ep_id = []
        self.safe2safe_ep_weight = None
        self.crash2crash_ep_id = []
        self.crash2crash_ep_weight = None
        self.crash2nearmiss_ep_id = []
        self.crash2nearmiss_ep_weight = None
        self.crash2safe_ep_id = []
        self.crash2safe_ep_weight = None
        self.sampled_crash = self.sampled_crash_still_crash = 0
        obs = self._reset()
        return obs

    def step(self, action):
        """Main function to step the environment.

        Args:
            action (np.array): Action to be taken.

        Returns:
            np.array: Observation after taking the action.
            float: Reward after taking the action.
            bool: Whether the episode is done.
            dict: Additional information.
        """
        if not conf.d2rl_flag:
            obs = self._get_observation()
            done, reason = self._get_done()
            self.cav_speed_list.append(self.env.vehicle_list["CAV"].observation.information["Ego"]["velocity"])
            self.cav_mean_speed = np.mean(self.cav_speed_list)
            self.obs_and_action.append([obs[0], action])
            if not done:
                self.total_step += 1
                self.simulator.step(action)
                # update for another time
                done, reason = self._get_done()
            else:
                pass
            reward = self._get_reward(done, reason)
            info = self._get_info()
        else:
            early_stop = False
            while 1:
                self.cav_speed_list.append(self.env.vehicle_list["CAV"].observation.information["Ego"]["velocity"])
                self.cav_mean_speed = np.mean(self.cav_speed_list)
                obs = self._get_observation()
                done, reason = self._get_done()
                if not done:
                    self.slightly_critical = self.env.vehicle_list["CAV"].controller.slightly_critical
                    self.highly_critical = self.env.vehicle_list["CAV"].controller.highly_critical
                    critical_condition = self.highly_critical
                    if conf.d2rl_slightlycritical:
                        critical_condition = self.highly_critical or self.slightly_critical
                    self.sub_critical_time_step_num += int(self.slightly_critical)
                    self.critical_time_step_num += int(self.highly_critical)
                    if critical_condition:
                        self.total_step += 1
                        self.simulator.step(action)
                        rss_info = self._get_critical_step_rss()
                        if rss_info["RSS_flag"] == 1:
                            self.critical_time_rss_num += 1
                        break
                    else:
                        self.total_step += 1
                        self.simulator.step()
                        if conf.experiment_config["ablation_study_config"] == "NRSMDP":
                            early_stop = True
                            break
                else:
                    break
            if done:
                # print("reason:", reason, "critical:", self.critical_time_step_num, "rss:", self.critical_time_rss_num)
                pass
            # update for another time
            obs = self._get_observation()
            done, reason = self._get_done()
            if early_stop:
                done = True
                reason = {9: "Replay ends"}
            reward = self._get_reward(done, reason)
            info = self._get_info()
        return obs, reward, done, info

    def fast_step(self):
        """Skip the steps which are not interested.

        Returns:
            np.array: Observation after skip the steps that are not interested.
            float: Reward after taking the action.
            bool: Whether the episode is done.
            dict: Additional information.
        """
        while 1:
            self.cav_speed_list.append(self.env.vehicle_list["CAV"].observation.information["Ego"]["velocity"])
            self.cav_mean_speed = np.mean(self.cav_speed_list)
            obs = self._get_observation()
            done, reason = self._get_done()
            if not done:
                self.slightly_critical = self.env.vehicle_list["CAV"].controller.slightly_critical
                self.highly_critical = self.env.vehicle_list["CAV"].controller.highly_critical
                self.total_step += 1
                self.simulator.step()
                if self.slightly_critical or self.highly_critical:
                    rss_info = self._get_critical_step_rss()
                    if rss_info["RSS_flag"] == 1:
                        self.critical_time_rss_num += 1
                # get to the interested time step
                if round(self.simulator.get_time()-0.1,1) == round(float(self.simulator.env.init_time_step),1):
                    break
            else:
                break
        if done:
            # print("reason:", reason, "critical:", self.critical_time_step_num, "rss:", self.critical_time_rss_num) 
            pass
        # update for another time
        obs = self._get_observation()
        done, reason = self._get_done()
        reward = self._get_reward(done, reason)
        info = self._get_info()
        return obs, reward, done, info

    def _get_observation(self):
        """Get the observation of the environment.

        Returns:
            np.array: Observation of the environment.
        """
        return self.env.vehicle_list["CAV"].controller.cav_obs_space
    
    def _get_criticality(self):
        """Get the criticality of the environment.

        Returns:
            float: Criticality of the environment.
        """
        if self.eval_flag:
            return conf.precise_criticality_threshold
        for global_controller in self.env.global_controller_instance_list:
            if global_controller.veh_type == "BV":
                if conf.env_mode == "NADE":
                    return np.sum(global_controller.original_NADE_decision.bv_criticality_list)
        return 0.0
    
    def _get_critical_step_rss(self):
        """Get the RSS information of the current step.

        Returns:
            dict: RSS information of the current step.
        """
        info = {"RSS_flag": 0, "additional_info": None}
        cav = self.env.vehicle_list["CAV"]
        if cav.controller.RSS_model.cav_veh.RSS_control:
            info["RSS_flag"] = 1
            info["additional_info"] = cav.controller.action
        return info

    def _get_reward(self, done, reason):
        """Get the reward of the current step.

        Args:
            done (bool): Whether the episode is done.
            reason (dict): Reason of ending the episode.

        Returns:
            float: Reward of the current step.
        """
        reward = 0
        normal_reward = 0.0 # 1
        crash_loss = -1 
        leave_loss = -0.2 # -1
        speed_reward_constant = 0 # 2e-4, +2e-3: The reward received when driving at 40 m/s, linearly mapped to zero for 35 m/s
        # speed_reward = np.clip(utils.remap(self.vehicle.velocity, [30, 40], [0, High_velocity_reward]), 0, High_velocity_reward)
        # current_cav_speed = self.vehicle_list["CAV"].observation.information["Ego"]["velocity"]
        # speed_reward = np.clip(utils.remap(current_cav_speed, [self.cav_agent.vel_low, self.cav_agent.vel_high], [-High_velocity_reward, High_velocity_reward]), -High_velocity_reward, High_velocity_reward)
        driving_distance_reward = 0.1
        lateral_offset_threshold = 0.2
        lanekeep_reward = 0 # 5e-4
        slow_loss = -0.2 # -1
        if not done:
            current_cav_speed = self.env.vehicle_list["CAV"].observation.information["Ego"]["velocity"]
            reward += speed_reward_constant*(current_cav_speed-20)/20.0
            cav_lateral_offset = abs(self.env.vehicle_list["CAV"].observation.information["Ego"]["lateral_offset"])
            reward += lanekeep_reward * (1-cav_lateral_offset)
            # reward = -speed_reward_constant
            self.reward_time += speed_reward_constant*(current_cav_speed-20)/20.0
            self.reward_lanekeep += lanekeep_reward * (1-cav_lateral_offset)
        else:
            # ep_id = self.simulator.input[0][0]
            if list(reason.keys()) == [1]:
                # crash case
                reward = crash_loss
                self.reward_crash = crash_loss
                if isinstance(self.data_info_origin, list):
                    data_source = self.data_info_origin[self.simulator.input[5]]
                else:
                    data_source = self.data_info_origin
                if self.simulator.input[0] in data_source["safe_id_list"]:
                    self.safe2crash_ep_id = list(self.simulator.input[0])
                    self.safe2crash_ep_weight = self.simulator.input[2]
                elif self.simulator.input[0] in data_source["crash_id_list"]:
                    self.crash2crash_ep_id = list(self.simulator.input[0])
                    self.crash2crash_ep_weight = self.simulator.input[2]
                if self.sampled_crash:
                    self.sampled_crash_still_crash = 1
            elif list(reason.keys()) == [4]:
                # normal experiment
                reward = normal_reward
                self.reward_normal = normal_reward
                if isinstance(self.data_info_origin, list):
                    data_source = self.data_info_origin[self.simulator.input[5]]
                else:
                    data_source = self.data_info_origin
                if self.simulator.input[0] in data_source["safe_id_list"]:
                    if self.env.three_circle_min_distance < conf.eval_three_circle_min_distance_threshold:
                        self.safe2nearmiss_ep_id = list(self.simulator.input[0])
                        self.safe2nearmiss_ep_weight = self.simulator.input[2]
                    else:
                        self.safe2safe_ep_id = list(self.simulator.input[0])
                        self.safe2safe_ep_weight = self.simulator.input[2]
                elif self.simulator.input[0] in data_source["crash_id_list"]:
                    if self.env.three_circle_min_distance < conf.eval_three_circle_min_distance_threshold:
                        self.crash2nearmiss_ep_id = list(self.simulator.input[0])
                        self.crash2nearmiss_ep_weight = self.simulator.input[2]
                    else:
                        self.crash2safe_ep_id = list(self.simulator.input[0])
                        self.crash2safe_ep_weight = self.simulator.input[2]
            elif list(reason.keys()) == [6]:
                reward = leave_loss
                self.reward_leave = leave_loss
            elif list(reason.keys()) == [7] or list(reason.keys()) == [8]:
                reward = slow_loss
                self.reward_spendrange = slow_loss
            elif list(reason.keys()) == [9]:
                reward = normal_reward
                self.reward_normal = normal_reward
                if isinstance(self.data_info_origin, list):
                    data_source = self.data_info_origin[self.simulator.input[5]]
                else:
                    data_source = self.data_info_origin
                if self.simulator.input[0] in data_source["safe_id_list"]:
                    if self.env.three_circle_min_distance < conf.eval_three_circle_min_distance_threshold:
                        self.safe2nearmiss_ep_id = list(self.simulator.input[0])
                        self.safe2nearmiss_ep_weight = self.simulator.input[2]
                    else:
                        self.safe2safe_ep_id = list(self.simulator.input[0])
                        self.safe2safe_ep_weight = self.simulator.input[2]
                elif self.simulator.input[0] in data_source["crash_id_list"]:
                    if self.env.three_circle_min_distance < conf.eval_three_circle_min_distance_threshold:
                        self.crash2nearmiss_ep_id = list(self.simulator.input[0])
                        self.crash2nearmiss_ep_weight = self.simulator.input[2]
                    else:
                        self.crash2safe_ep_id = list(self.simulator.input[0])
                        self.crash2safe_ep_weight = self.simulator.input[2]
        return reward

    def _get_done(self):
        """Get the done flag and reason of ending the episode.

        Returns:
            bool: Whether the episode is done.
            dict: Reason of ending the episode.
        """
        stop, reason, _ = self.env.terminate_check()
        return stop, reason
    
    def _get_info(self):
        """Get the additional information of the environment.

        Returns:
            dict: Additional information of the environment.
        """
        if conf.env_mode == "NADE":
            weight = self.simulator.input[2]
        else:
            weight = 1
        if isinstance(self.data_info_origin, list):
            traj_pool_size = []
            for i in range(len(self.tmp_crash_id_list)):
                traj_pool_size.append(len(self.tmp_crash_id_list[i]))
                traj_pool_size.append(len(self.tmp_safe_id_list[i]))
        else:
            traj_pool_size = [len(self.tmp_crash_id_list), len(self.tmp_safe_id_list)]
        info = {
            "CAV_mean_speed": self.cav_mean_speed,
            "reward_time": self.reward_time,
            "reward_crash": self.reward_crash,
            "reward_normal": self.reward_normal,
            "reward_speedrange": self.reward_spendrange,
            "reward_leave": self.reward_leave,
            "reward_lanekeep": self.reward_lanekeep, 
            "sub_critical_time_step_number": self.sub_critical_time_step_num,
            "critical_time_step_number": self.critical_time_step_num,
            "critical_time_rss_number": self.critical_time_rss_num,
            "sub_critical_time_step_number": self.sub_critical_time_step_num,
            "total_episode_number": self.total_episode,
            "total_step_number": self.total_step,
            "episode_weight": weight,
            "episode_len": self.env.simulator.sumo_time_stamp,
            "highly_critical": self.highly_critical,
            "slightly_critical": self.slightly_critical,
            "safe2crash_id": self.safe2crash_ep_id,
            "safe2crash_weight": 0 if self.safe2crash_ep_weight is None else self.safe2crash_ep_weight,
            "safe2nearmiss_id": self.safe2nearmiss_ep_id,
            "safe2nearmiss_weight": 0 if self.safe2nearmiss_ep_weight is None else self.safe2nearmiss_ep_weight,
            "safe2safe_id": self.safe2safe_ep_id,
            "safe2safe_weight": 0 if self.safe2safe_ep_weight is None else self.safe2safe_ep_weight,
            "crash2crash_id": self.crash2crash_ep_id,
            "crash2crash_weight": 0 if self.crash2crash_ep_weight is None else self.crash2crash_ep_weight,
            "crash2nearmiss_id": self.crash2nearmiss_ep_id,
            "crash2nearmiss_weight": 0 if self.crash2nearmiss_ep_weight is None else self.crash2nearmiss_ep_weight,
            "crash2safe_id": self.crash2safe_ep_id,
            "crash2safe_weight": 0 if self.crash2safe_ep_weight is None else self.crash2safe_ep_weight,
            "data_folder_index": 0 if self.data_folder_index is None else self.data_folder_index,
            "three_circle_min_distance": self.env.three_circle_min_distance,
            "traj_pool_size": traj_pool_size,
            "interest_time": self.env.soft_reboot_time_section,
            "interest_time2": time.time()-self.init_wall_time,
            "interested_crash_sampling_stats": [self.sampled_crash, self.sampled_crash_still_crash, self.data_folder_index],
            "ep_id": self.ep_id
        }
        return info

    def close(self):
        """Close the environment. To be more specific, stop the simulator.
        """
        self.simulator.stop()

    def set_eval_index(self):
        """Set the index for the evaluation worker.
        """
        # self.load_offline_dataset()
        self.eval_flag = True
        self.eval_index = 0

    def set_traj_pool(self, tmp_safe_id_list, tmp_safe_weight, tmp_crash_id_list, tmp_crash_weight, num_crash=0):
        """Set the trajectory pool for different evaluation workers.
        """
        self.tmp_safe_id_list = tmp_safe_id_list
        self.tmp_safe_weight = tmp_safe_weight
        self.tmp_crash_id_list = tmp_crash_id_list
        self.tmp_crash_weight = tmp_crash_weight
        self.num_crash = num_crash

    def load_offline_dataset(self):
        """Load the offline dataset for the evaluation workers.

        Returns:
            float: Time used for loading the offline dataset.
            list: Index of the loaded files to replay and collect data.
        """
        if self.loaded_safe_files_index is not None:
            return [0.0, self.loaded_safe_files_index]
        t0 = time.time()
        safe_folder = os.path.join(conf.data_folder[0], "tested_and_safe")
        safe_fcd_files = sorted(glob.glob(safe_folder+"/*.fcd.json"))
        worker_index = int(self.eval_worker_index.split("/")[0])
        num_worker = int(self.eval_worker_index.split("/")[-1])
        num_files = int(len(safe_fcd_files)/float(num_worker))
        if worker_index == num_worker:
            loaded_safe_fcd_files = safe_fcd_files[(worker_index-1)*num_files:]
        else:
            loaded_safe_fcd_files = safe_fcd_files[(worker_index-1)*num_files:(worker_index)*num_files]
        self.loaded_safe_files_index = [int(name.split("/")[-1].replace(".fcd.json","")) for name in loaded_safe_fcd_files]
        self.fcd_trajs = {}
        self.json_trajs = {}
        for fcd_file in loaded_safe_fcd_files:
            with open(fcd_file) as fcd_obj:
                for line1 in fcd_obj:
                    fcd_iter_info = ujson.loads(line1)
                    self.fcd_trajs[int(fcd_iter_info["original_name"])] = json2dict(fcd_iter_info)
            json_file = fcd_file.replace(".fcd.json",".json")
            with open(json_file) as json_obj:
                for line2 in json_obj:
                    json_iter_info = ujson.loads(line2)
                    self.json_trajs[int(json_iter_info["episode_info"]["id"])] = json_iter_info
        output = [time.time()-t0, self.loaded_safe_files_index]
        print(f"load offline dataset for worker{worker_index}", output)
        return output

    def get_eval_trajs(self, eval_index_list, tmp_safe_id_list):
        """Get the evaluation trajectories for the evaluation worker.

        Args:
            eval_index_list (list): Index of the trajectories to be evaluated.
            tmp_safe_id_list (list): Information of all the safe trajectories.

        Returns:
            list: Evaluation trajectories.
        """
        trajs_list = []
        for i in eval_index_list:
            if tmp_safe_id_list[i][0] in self.fcd_trajs and tmp_safe_id_list[i][0] in self.json_trajs:
                trajs_list.append([
                    self.fcd_trajs[tmp_safe_id_list[i][0]], 
                    self.json_trajs[tmp_safe_id_list[i][0]],
                    tmp_safe_id_list[i]
                ])
        return trajs_list

    def set_eval_trajs(self, eval_index_list_eval_worker):
        """Set the evaluation trajectories for the evaluation worker.

        Args:
            eval_index_list_eval_worker (list): Index of the trajectories to be evaluated.
        """
        worker_index = int(self.worker_index.split("/")[0])
        self.eval_index_list = list(eval_index_list_eval_worker[worker_index-1])

    def set_eval_trajs_new(self, whole_traj_pool):
        """Set the evaluation trajectories for the evaluation worker to be all the trajectories.

        Args:
            whole_traj_pool (list): Information of all the trajectories.
        """
        # worker_index = int(self.eval_worker_index.split("/")[0])
        # num_worker = int(self.eval_worker_index.split("/")[-1])
        # num_trajs = int(len(whole_traj_pool)/float(num_worker))
        # if worker_index == num_worker:
        #     self.eval_trajs = whole_traj_pool[worker_index*num_trajs:]
        # else:
        #     self.eval_trajs = whole_traj_pool[worker_index*num_trajs:(worker_index+1)*num_trajs]
        self.eval_trajs = whole_traj_pool