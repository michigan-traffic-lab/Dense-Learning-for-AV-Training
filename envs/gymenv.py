from gym import core
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import math

from mtlsp.simulator import Simulator
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


class RL_NDE(core.Env):
    def __init__(self, train_flag=True):
        """Initialize the RL_NDE class.

        Args:
            train_flag (bool): Whether the environment is used for training.
        """
        self.env = None
        self.simulator = None
        # Actions should include 31 accelerations (-4:0.2:2 m/s^2) and 10 steering angles (-0.5:0.1:-0.1, 0.1:0.1:0.5 deg) 
        self.action_space = Box(low=np.array([-4,-10]), high=np.array([2,10]), shape=(2, ))
        self.observation_space = Box(low=-1, high=1, shape=(43, ))
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
        self.small_weight_step_number = 0
        self.small_weight_step_list = []
        self.small_weight_list = []
        self.weight_step_index = 0
        self.reset_flag = False

    def _reset(self):
        """Helper function to reset the environment.

        Returns:
            np.array: The initial observation of the environment.
        """
        self.total_episode += 1
        # reset environment
        if self.simulator is not None:
            self.simulator.stop()
        sumo_config_file_path = "/maps/3LaneHighway/3LaneHighway.sumocfg"
        if conf.env_mode == "NDE":
            self.env = NDE(AVController=RLControllerNew)
        elif conf.env_mode == "NADE":
            self.env = NADE(BVController=TreeSearchNADEBackgroundController, cav_model="RLNew")
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
			output=[],
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
            self.slightly_critical = self.env.vehicle_list["CAV"].controller.slightly_critical
            self.highly_critical = self.env.vehicle_list["CAV"].controller.highly_critical
            critical_condition = self.highly_critical
            if conf.d2rl_slightlycritical:
                critical_condition = self.highly_critical or self.slightly_critical
            if conf.d2rl_flag:
                if not critical_condition:
                    obs, reward, done, info = self.step(np.array([0,0]))
                    if done:
                        return self._reset()
        return obs

    def reset(self):
        """Main function to reset the environment.

        Returns:
            np.array: The initial observation of the environment.
        """
        # reset parameters
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
        self.small_weight_step_number = 0
        self.small_weight_step_list = []
        self.small_weight_list = []
        self.weight_step_index = 0
        self.reset_flag = True
        obs = self._reset()
        return obs

    def step(self, action):
        """Main function to take a step in the environment.

        Args:
            action (np.array): The action to be taken in the environment.

        Returns:
            np.array: The observation after taking the action.
            float: The reward after taking the action.
            bool: Whether the episode is done.
            dict: Additional information about the episode.
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
            while 1:
                step_flag = False
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
                    if critical_condition:
                        if not self.reset_flag:
                            self.total_step += 1
                            self.critical_time_step_num += 1
                            step_flag = True
                            self.simulator.step(action)
                            rss_info = self._get_critical_step_rss()
                            if rss_info["RSS_flag"] == 1:
                                self.critical_time_rss_num += 1
                        else:
                            self.reset_flag = False
                        break
                    else:
                        self.total_step += 1
                        self.simulator.step()
                else:
                    break
            if done:
                # print("reason:", reason, "critical:", self.critical_time_step_num, "rss:", self.critical_time_rss_num) 
                pass
            # update for another time
            done, reason = self._get_done()
            reward = self._get_reward(done, reason)
            info = self._get_info(done, step_flag)
        return obs, reward, done, info

    def _get_observation(self):
        """Helper function to get the observation of the environment.

        Returns:
            np.array: The observation of the environment.
        """
        return self.env.vehicle_list["CAV"].controller.cav_obs_space
    
    def _get_criticality(self):
        """Helper function to get the criticality of the environment.

        Returns:
            float: The criticality of the environment.
        """
        for global_controller in self.env.global_controller_instance_list:
            if global_controller.veh_type == "BV":
                if conf.env_mode == "NADE":
                    return np.sum(global_controller.original_NADE_decision.bv_criticality_list)
        return 0.0
    
    def _get_critical_step_rss(self):
        """Helper function to get the RSS information of the environment.

        Returns:
            dict: The RSS information of the environment.
        """
        info = {"RSS_flag": 0, "additional_info": None}
        cav = self.env.vehicle_list["CAV"]
        if cav.controller.RSS_model.cav_veh.RSS_control:
            info["RSS_flag"] = 1
            info["additional_info"] = cav.controller.action
        return info

    def _get_reward(self, done, reason):
        """Helper function to get the reward of the environment.

        Args:
            done (bool): Whether the episode is done.
            reason (dict): The reason for the episode to be done.

        Returns:
            float: The reward of the environment.
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
            if list(reason.keys()) == [1]:
                # crash case
                reward = crash_loss
                self.reward_crash = crash_loss
            elif list(reason.keys()) == [4]:
                # normal experiment
                reward = normal_reward
                self.reward_normal = normal_reward
            elif list(reason.keys()) == [6]:
                reward = leave_loss
                self.reward_leave = leave_loss
            elif list(reason.keys()) == [7] or list(reason.keys()) == [8]:
                reward = slow_loss
                self.reward_spendrange = slow_loss
        return reward

    def _get_done(self):
        """Helper function to check if the episode is done.

        Returns:
            bool: Whether the episode is done.
            dict: The reason for the episode to be done.
        """
        stop, reason, _ = self.env.terminate_check()
        return stop, reason
    
    def _get_info(self, done_flag, step_flag):
        """Helper function to get the information of the environment.

        Args:
            done_flag (bool): Whether the episode is done.
            step_flag (bool): Whether the step is studied.

        Returns:
            dict: The information of the environment.
        """
        if conf.env_mode == "NADE":
            episode_weight = self.env.info_extractor.episode_log["weight_episode"]
            current_weight = self.env.info_extractor.episode_log["current_weight"]
        else:
            episode_weight = 1
            current_weight = 1
        if step_flag:
            self.weight_step_index += 1
            if not math.isclose(episode_weight, 1) and episode_weight < 1:
                self.small_weight_step_number += 1
                self.small_weight_step_list.append(self.weight_step_index)
                self.small_weight_list.append(episode_weight)
        if done_flag:
            if self.reward_crash == -1:
                normal_small_weight_index_list, normal_small_weight_list = [], []
                crash_small_weight_index_list, crash_small_weight_list = self.small_weight_step_list, self.small_weight_list
            else:
                normal_small_weight_index_list, normal_small_weight_list = self.small_weight_step_list, self.small_weight_list
                crash_small_weight_index_list, crash_small_weight_list = [], []
        else:
            normal_small_weight_index_list, normal_small_weight_list = [], []
            crash_small_weight_index_list, crash_small_weight_list = [], []

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
            "episode_weight": episode_weight,
            "current_weight": current_weight,
            "episode_len": self.env.simulator.sumo_time_stamp,
            "highly_critical": self.highly_critical,
            "slightly_critical": self.slightly_critical,
            "small_weight_step_number": self.small_weight_step_number,
            "normal_small_weight_index_list": normal_small_weight_index_list,
            "crash_small_weight_index_list": crash_small_weight_index_list,
            "normal_small_weight_list": normal_small_weight_list,
            "crash_small_weight_list": crash_small_weight_list
        }
        return info

    def close(self):
        """Helper function to close the environment.
        """
        self.simulator.stop()

