import numpy as np
from mtlsp.simulator import Simulator
from mtlsp.controller.vehicle_controller.controller import Controller, BaseController
import mtlsp.utils as utils
import os
if os.environ["mode"] == "testing":
    import conf.conf_testing as conf
elif os.environ["mode"] == "training":
    import conf.conf_training as conf
else:
    raise ValueError("Please set the mode to testing or training")
from utils import acceleration
from .RSS_model import highway_RSS_model
from .nadeglobalcontroller import NADEBVGlobalController
from .nddcontroller import NDDController
from .treesearchnadecontroller import TreeSearchNADEBackgroundController
from envs.nde import NDE
from mtlsp.utils import update_vehicle_real_states
from scipy.optimize import fsolve
import time
import collections
from .neuralmetric import NN_Metric

class RLController(BaseController):
    def __init__(self, subscription_method=Simulator.subscribe_vehicle_all_information, controllertype="RLController"):
        """Initialize the RLController class.

        Args:
            subscription_method (function, optional): Subscription method. Defaults to Simulator.subscribe_vehicle_all_information.
            controllertype (str, optional): Type of the controller. Defaults to "RLController".
        """
        super().__init__(subscription_method=subscription_method, controllertype=controllertype)
        self.state = None

    def install(self):
        """Install vehicle controller and combine training agent.
        """        
        super().install()
        self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_red)
        self.agent = self.vehicle.simulator.env.cav_agent

    def step(self):
        """Decide the AV maneuver in the next step.
        """        
        super().step()
        if not conf.experiment_config["train_flag"]:
            action_indicator = self.get_action_indicator(self.vehicle.observation, ndd_flag = False, safety_flag = True, CAV_flag = True)
            action_indicator_after_lane_conflict = self.lane_conflict_safety_check(self.vehicle.observation, action_indicator)
        else:
            action_indicator_after_lane_conflict = self.get_action_indicator(self.vehicle.observation, ndd_flag = False, safety_flag = False, CAV_flag = True)
            assert(action_indicator_after_lane_conflict.all())
        s, action_index = self.agent.decision(self.vehicle.observation, action_indicator_after_lane_conflict, evaluation_flag=False)
        actions = {}
        if action_index <= 1:
            actions["lateral"] = self.agent.ACTIONS[action_index]
            actions["longitudinal"] = 0.0
        else:
            actions["lateral"] = "central"
            actions["longitudinal"] = float(self.agent.ACTIONS[action_index])
        self.action = actions
        self.action_index = action_index
        self.state = s
        self.action_indicator = action_indicator_after_lane_conflict

    def get_action_indicator(self, observation, ndd_flag = False, safety_flag = True, CAV_flag = False):
        """Get the action indicator for the vehicle.

        Args:
            observation (Observation): Observation of the vehicle.
            ndd_flag (bool, optional): Check whether the vehicle follows the NDD model. Defaults to False.
            safety_flag (bool, optional): Check whether safety check is necessary. Defaults to True.
            CAV_flag (bool, optional): Check whether the vehicle is CAV. Defaults to False.

        Raises:
            ValueError: If this function is called by BV, raise error.

        Returns:
            list(bool): Action indicator showing whether each action is safe.
        """        
        if CAV_flag:
            action_shape = len(self.agent.ACTIONS)
            ndd_action_indicator = np.ones(action_shape)
            if ndd_flag:
                pass
            safety_action_indicator = np.ones(action_shape)
            if safety_flag:
                lateral_action_indicator = np.array([1, 1, 1])
                lateral_result = BaseController._check_lateral_safety(observation.information, lateral_action_indicator, CAV_flag=True)
                longi_result = BaseController._check_longitudinal_safety(observation.information, np.ones(action_shape-2), lateral_result=lateral_result, CAV_flag=True)
                safety_action_indicator[0], safety_action_indicator[1] = lateral_result[0], lateral_result[2]      
                safety_action_indicator[2:] = longi_result                
            action_indicator = ndd_action_indicator * safety_action_indicator
            action_indicator = (action_indicator > 0)
            return action_indicator

        else:
            raise ValueError("Get BV action Indicator in CAV function")

    def lane_conflict_safety_check(self, observation, action_indicator_before):
        """Check lane_conflict_safety safety.

        Args:
            action_indicator_before (list(bool)): List of old action indicator.

        Returns:
            list(bool): New action indicator list after checking lane conflict case.
        """        
        # If there is no longitudinal actions are OK or in the middle lane, then do not block the lane change probability
        CAV_info = observation.information["Ego"]
        CAV_v, CAV_x, CAV_current_lane_id = CAV_info["velocity"], CAV_info["position"][0], CAV_info["lane_index"]
        if (not action_indicator_before[2:].any()) or (CAV_current_lane_id == 1):
            return action_indicator_before

        # If there is no lane change probability, just return
        if (CAV_current_lane_id==0 and not action_indicator_before[0]) or (CAV_current_lane_id==2 and not action_indicator_before[1]):
            return action_indicator_before

        if CAV_current_lane_id == 0: candidate_BV_lane, CAV_ban_lane_change_id = 2, 0
        elif CAV_current_lane_id == 2: candidate_BV_lane, CAV_ban_lane_change_id = 0, 1
        # candidate_BV
        bvs = observation.context
        candidate_BV_dict = {}
        for veh_id in bvs.keys():
            if bvs[veh_id][82] == candidate_BV_lane:
                candidate_BV_dict[veh_id] = bvs[veh_id]
        if len(candidate_BV_dict) == 0:
            return action_indicator_before

        r_now, rr_now, r_1_second, r_2_second = [], [], [], []
        for veh_id in candidate_BV_dict.keys():
            BV_x, BV_v = candidate_BV_dict[veh_id][66][0], candidate_BV_dict[veh_id][64]
            if BV_x >= CAV_x:
                r_now_tmp = BV_x - CAV_x - self.LENGTH
                rr_now_tmp = BV_v - CAV_v
                r_1_second_tmp = r_now_tmp + rr_now_tmp*conf.simulation_resolution
                acc_BV = acc_CAV = self.acc_low
                BV_dis = utils.cal_dis_with_start_end_speed(BV_v, np.clip(BV_v+acc_BV, self.v_low, self.v_high), acc_BV, time_interval=conf.simulation_resolution, v_low=self.v_low, v_high=self.v_high)
                CAV_dis = utils.cal_dis_with_start_end_speed(CAV_v, np.clip(CAV_v+acc_CAV, self.v_low, self.v_high), acc_CAV, time_interval=conf.simulation_resolution, v_low=self.v_low, v_high=self.v_high)
                r_2_second_tmp = r_1_second_tmp + BV_dis - CAV_dis
                r_now.append(r_now_tmp)
                rr_now.append(rr_now_tmp)
                r_1_second.append(r_1_second_tmp)
                r_2_second.append(r_2_second_tmp)
            else:
                r_now_tmp = CAV_x - BV_x - self.LENGTH
                rr_now_tmp = CAV_v - BV_v
                r_1_second_tmp = r_now_tmp + rr_now_tmp*conf.simulation_resolution
                acc_BV = self.acc_low
                acc_CAV = 0
                BV_dis = utils.cal_dis_with_start_end_speed(BV_v, np.clip(BV_v+acc_BV, self.v_low, self.v_high), acc_BV, time_interval=conf.simulation_resolution, v_low=self.v_low, v_high=self.v_high)
                CAV_dis = utils.cal_dis_with_start_end_speed(CAV_v, np.clip(CAV_v+acc_CAV, self.v_low, self.v_high), acc_CAV, time_interval=conf.simulation_resolution, v_low=self.v_low, v_high=self.v_high)            
                r_2_second_tmp = r_1_second_tmp + CAV_dis - BV_dis
                r_now.append(r_now_tmp)
                rr_now.append(rr_now_tmp)
                r_1_second.append(r_1_second_tmp)
                r_2_second.append(r_2_second_tmp)
        r_now, r_1_second, r_2_second = np.array(r_now), np.array(r_1_second), np.array(r_2_second)
        if (r_now <= 0).any() or (r_1_second <= 0).any() or (r_2_second <= 0).any():
            # Sample to decide whether ban the lane change
            if np.random.rand() <= conf.ignore_lane_conflict_prob:
                return action_indicator_before
                
            else:
                action_indicator_before[CAV_ban_lane_change_id] = False
                return action_indicator_before
    
        return action_indicator_before


class RLControllerNew(BaseController):
    veh_length = 5.0
    road_geo = [[40.0, 44.0], [44.0, 48.0], [48.0, 52.0]]
    road_center = [42.0, 46.0, 50.0]
    considered_bv_num = 4
    acc_low, acc_high, acc_res = -4, 2, 0.2 # 31 accelerations
    angle_max, angle_res = 10, 0.1 # 10 angles
    actions = []
    for i in range(31):
        acc = round(acc_low+acc_res*i, 1)
        actions.append([acc, 0.0])
    for i in range(11):
        angle = -angle_max+angle_res*i
        if angle != 0:
            actions.append([0, angle])

    def __init__(self, subscription_method=Simulator.subscribe_vehicle_all_information, controllertype="RLControllerNew"):
        """Initialize the RLControllerNew class.

        Args:
            subscription_method (function, optional): Subscription method. Defaults to Simulator.subscribe_vehicle_all_information.
            controllertype (str, optional): Type of the controller. Defaults to "RLControllerNew".
        """
        super().__init__(subscription_method=subscription_method, controllertype=controllertype)
        self.RSS_model = highway_RSS_model()
        if conf.simulation_config["neuralmetric_flag"]:
            self.NN_metric = NN_Metric(conf.simulation_config["neuralmetric_config"])
        else:
            self.NN_metric = None
        self._recent_critical = None
        self._recent_cav_obs_space = None

    def install(self):
        """Install vehicle controller and combine training agent.
        """        
        super().install()
        self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_red)
        if self.vehicle.simulator.input is not None:
            self.vehicle.states = list(self.vehicle.simulator.env.cav_initial_states)
            self.vehicle.lateral_speed = self.vehicle.simulator.env.cav_initial_lateral_speed
            self.RSS_model.update_states(self.vehicle.simulator.env.cav_initial_RSS_states, self.vehicle.simulator.env.init_time_step)

    # @profile
    def step(self):
        """Decide the AV maneuver in the next step.
        """    
        if conf.computational_analysis_flag:
            t0 = time.time()    
        super().step()
        self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_red)
        obs = self.cav_obs_space
        obs_NNMetric = self.cav_neuralmetric_obs_space
        if conf.computational_analysis_flag:
            t1 = time.time()   
            print("av obs (ms):", (t1-t0)*1000) 
        
        acc_lat_max = 4
        if self.vehicle.simulator.env.cav_action is not None:
            action = self.vehicle.simulator.env.cav_action
            if conf.RSS_flag:
                new_states = update_vehicle_real_states(self.vehicle.states, {"acceleration": float(action[0]), "steering_angle": float(action[1])}, self.vehicle.params, self.vehicle.step_size)
                dx = new_states[1] - self.vehicle.states[1]
                current_v = self.vehicle.observation.information["Ego"]["velocity"]
                new_v = dx/self.vehicle.step_size*2-current_v
                acc = (new_v-current_v)/self.vehicle.step_size
                dy = new_states[2] - self.vehicle.states[2]
                new_lateral_v = dy/self.vehicle.step_size*2-self.vehicle.lateral_speed
                acc_lat =(new_lateral_v-self.vehicle.lateral_speed)/self.vehicle.step_size
                RSS_restriction = self.RSS_model.RSS_act_CAV(self.vehicle.simulator.env, {"acc_x":acc, "acc_y":acc_lat})
                RSS_control = self.RSS_model.RSS_step_CAV(env=self.vehicle.simulator.env)
                # print("RSS check:",time.time()-t0)
                # if not self.RSS_model.cav_veh.RSS_control:
                if not self.RSS_model.cav_veh.RSS_control or self.highly_critical:
                    # check acceleration limit
                    if acc > conf.acc_high or acc < conf.acc_low or abs(acc_lat) > acc_lat_max:
                        acc_dict = {"acc_x":np.clip(acc,conf.acc_low,conf.acc_high),"acc_y":np.clip(acc_lat,-acc_lat_max,acc_lat_max)}
                        action_list = fsolve(self.helper_solve_for_control_inputs, [float(action[0]), float(action[1])], args=(acc_dict,))
                    else:
                        action_list = [float(action[0]), float(action[1])]
                    self.action = {
                        "acceleration": np.clip(float(action_list[0]), self.acc_low, self.acc_high),
                        "steering_angle": np.clip(float(action_list[1]), -self.angle_max, self.angle_max),
                        "additional_info": [list(obs.astype("float64"))]
                    }
                else:
                    # self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_orange)
                    RSS_action = self.RSS_model.cav_veh.action
                    action_list = fsolve(self.helper_solve_for_control_inputs, [float(action[0]), float(action[1])], args=(RSS_action,))
                    self.action = {
                        "acceleration": np.clip(float(action_list[0]), self.acc_low, self.acc_high),
                        "steering_angle": np.clip(float(action_list[1]), -self.angle_max, self.angle_max),
                        "additional_info": [list(obs.astype("float64")), RSS_restriction["CAV"], RSS_control]
                    }
            else:
                if acc > conf.acc_high or acc < conf.acc_low or abs(acc_lat) > acc_lat_max:
                    acc_dict = {"acc_x":np.clip(acc,conf.acc_low,conf.acc_high),"acc_y":np.clip(acc_lat,-acc_lat_max,acc_lat_max)}
                    action_list = fsolve(self.helper_solve_for_control_inputs, [float(action[0]), float(action[1])], args=(acc_dict,))
                else:
                    action_list = [float(action[0]), float(action[1])]
                self.action = {
                    "acceleration": np.clip(float(action_list[0]), self.acc_low, self.acc_high),
                    "steering_angle": np.clip(float(action_list[1]), -self.angle_max, self.angle_max),
                    "additional_info": [list(obs.astype("float64"))]
                }
        else:
            if conf.cav_agent is None:
                if conf.simulation_config["pytorch_model_path_list"] != []:
                    conf.cav_agent = conf.load_ray_agent(conf.simulation_config["pytorch_model_path_list"])
            # assert(conf.cav_agent is not None)
            # t0 = time.time()
            if len(conf.cav_agent.agent_list) == 2 and self.highly_critical:
                # self.highly_critical = True
                self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_purple)
            elif len(conf.cav_agent.agent_list) == 3:
                if self.highly_critical:
                    self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_purple)
                elif self.slightly_critical:
                    self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_yellow)
            else:
                # self.highly_critical = False
                pass
            if conf.computational_analysis_flag:
                t2 = time.time()
                print("av criticality (ms):", (t2-t1)*1000)

            if not conf.d2rl_flag:
                action = conf.cav_agent.compute_action(obs, self.highly_critical, self.slightly_critical)
            else:
                # if self.highly_critical or self.slightly_critical:
                #     print("find bug!")
                action = conf.cav_agent.compute_action(obs, self.highly_critical, self.slightly_critical)
            if conf.computational_analysis_flag:
                t3 = time.time()
                print("av compute action (ms):", (t3-t2)*1000)

            new_states = update_vehicle_real_states(self.vehicle.states, {"acceleration": float(action[0]), "steering_angle": float(action[1])}, self.vehicle.params, self.vehicle.step_size)
            dx = new_states[1] - self.vehicle.states[1]
            current_v = self.vehicle.observation.information["Ego"]["velocity"]
            new_v = dx/self.vehicle.step_size*2-current_v
            acc = (new_v-current_v)/self.vehicle.step_size
            dy = new_states[2] - self.vehicle.states[2]
            new_lateral_v = dy/self.vehicle.step_size*2-self.vehicle.lateral_speed
            acc_lat =(new_lateral_v-self.vehicle.lateral_speed)/self.vehicle.step_size

            # if conf.RSS_flag and not self.highly_critical:
            if conf.RSS_flag:
                # print("agent decision:",time.time()-t0)
                RSS_restriction = self.RSS_model.RSS_act_CAV(self.vehicle.simulator.env, {"acc_x":acc, "acc_y":acc_lat})
                RSS_control = self.RSS_model.RSS_step_CAV(env=self.vehicle.simulator.env)
                new_RSS_state_dict = {}
                for pair in self.RSS_model.RSS_state_dict:
                    assert(pair[0]=="CAV")
                    new_RSS_state_dict[pair[1]] = dict(self.RSS_model.RSS_state_dict[pair])
                # print("RSS check:",time.time()-t0)
                if conf.simulation_config["safetyguard_flag"]:
                    RSS_excution_flag = self.RSS_model.cav_veh.RSS_control and not self.highly_critical
                else:
                    RSS_excution_flag = self.RSS_model.cav_veh.RSS_control
                if not RSS_excution_flag:
                    # consider acceleration limit
                    if acc > conf.acc_high or acc < conf.acc_low or abs(acc_lat) > acc_lat_max:
                        acc_dict = {"acc_x":np.clip(acc,conf.acc_low,conf.acc_high),"acc_y":np.clip(acc_lat,-acc_lat_max,acc_lat_max)}
                        action_list = fsolve(self.helper_solve_for_control_inputs, [float(action[0]), float(action[1])], args=(acc_dict,))
                    else:
                        action_list = [float(action[0]), float(action[1])]
                    self.action = {
                        "acceleration": np.clip(float(action_list[0]), self.acc_low, self.acc_high),
                        "steering_angle": np.clip(float(action_list[1]), -self.angle_max, self.angle_max),
                        "additional_info": {
                                "rl_obs": list(obs.astype("float64")),
                                "veh_states": list(self.vehicle.states),
                                "RSS_states": new_RSS_state_dict,
                                "NN_metric_obs": obs_NNMetric.astype("float64").tolist()[0],
                            }
                    }
                else:
                    RSS_action = self.RSS_model.cav_veh.action
                    action_list = fsolve(self.helper_solve_for_control_inputs, [float(action[0]), float(action[1])], args=(RSS_action,))
                    self.action = {
                        "acceleration": np.clip(float(action_list[0]), self.acc_low, self.acc_high),
                        "steering_angle": np.clip(float(action_list[1]), -self.angle_max, self.angle_max),
                        "additional_info": {
                            "rl_obs": list(obs.astype("float64")),
                            "veh_states": list(self.vehicle.states),
                            "RSS_states": new_RSS_state_dict,
                            "RSS_restrict": RSS_restriction["CAV"], 
                            "RSS_control": RSS_control,
                            "original_action": list(action.astype("float64")), 
                            "NN_metric_obs": obs_NNMetric.astype("float64").tolist()[0],
                        }
                    }
                    # print("Get control:", time.time()-t0)
                    # print(RSS_action, self.action, self.helper_solve_for_control_inputs(action_list, RSS_action))
            else:
                new_RSS_state_dict = {}
                for pair in self.RSS_model.RSS_state_dict:
                    assert(pair[0]=="CAV")
                    new_RSS_state_dict[pair[1]] = dict(self.RSS_model.RSS_state_dict[pair])
                # consider acceleration limit
                if acc > conf.acc_high or acc < conf.acc_low or abs(acc_lat) > acc_lat_max:
                    acc_dict = {"acc_x":np.clip(acc,conf.acc_low,conf.acc_high),"acc_y":np.clip(acc_lat,-acc_lat_max,acc_lat_max)}
                    action_list = fsolve(self.helper_solve_for_control_inputs, [float(action[0]), float(action[1])], args=(acc_dict,))
                else:
                    action_list = [float(action[0]), float(action[1])]
                self.action = {
                    "acceleration": float(action_list[0]),
                    "steering_angle": float(action_list[1]),
                    "additional_info": {
                        "rl_obs": list(obs.astype("float64")),
                        "veh_states": list(self.vehicle.states),
                        "RSS_states": new_RSS_state_dict,
                        "NN_metric_obs": obs_NNMetric.astype("float64").tolist()[0],
                    }
                }
            if conf.computational_analysis_flag:
                t4 = time.time()
                print("av rss action (ms):", (t4-t3)*1000)
        # else:
        #     print("find")

    @property
    def cav_obs_space(self):
        """Get the observation space of the CAV.

        Returns:
            np.array: Observation space of the CAV.
        """
        if self._recent_cav_obs_space is None or self.vehicle.simulator.get_time() != self._recent_cav_obs_space["time_step"]:
            self._recent_cav_obs_space = {
                "obs": self.vehicle.simulator.env.get_observation_drl(),
                "time_step": self.vehicle.simulator.get_time(),
                "neuralmetric_obs": np.array([-1]),
            }
            if self.NN_metric is not None:
                if self.vehicle.simulator.sumo_time_stamp == 0 and conf.train_mode == "offline":
                    self._recent_cav_obs_space["neuralmetric_obs"] = self.vehicle.simulator.env.get_observation_neuralmetric()
                else:
                    self._recent_cav_obs_space["neuralmetric_obs"] = self.NN_metric.normalize(self.vehicle.simulator.env.get_observation_neuralmetric())
        return self._recent_cav_obs_space["obs"]

    @property
    def cav_neuralmetric_obs_space(self):
        """Get the neural metric observation space of the CAV.

        Returns:
            np.array: Neural metric observation space of the CAV.
        """
        if self._recent_cav_obs_space is None or self.vehicle.simulator.get_time() != self._recent_cav_obs_space["time_step"]:
            self._recent_cav_obs_space = {
                "obs": self.vehicle.simulator.env.get_observation_drl(),
                "time_step": self.vehicle.simulator.get_time(),
                "neuralmetric_obs": np.array([-1]),
            }
            if self.NN_metric is not None:
                if self.vehicle.simulator.sumo_time_stamp == 0 and conf.train_mode == "offline":
                    self._recent_cav_obs_space["neuralmetric_obs"] = self.vehicle.simulator.env.get_observation_neuralmetric()
                else:
                    self._recent_cav_obs_space["neuralmetric_obs"] = self.NN_metric.normalize(self.vehicle.simulator.env.get_observation_neuralmetric())
        return self._recent_cav_obs_space["neuralmetric_obs"]

    @property
    def highly_critical(self):
        """Get the highly critical flag of the CAV.

        Returns:
            bool: Highly critical flag of the CAV.
        """
        if self._recent_critical is None or self.vehicle.simulator.get_time() != self._recent_critical["time_step"]:
            self._recent_critical = self.update_critical()
        return self._recent_critical["highly_critical"]

    @property
    def slightly_critical(self):
        """Get the slightly critical flag of the CAV.

        Returns:
            bool: Slightly critical flag of the CAV.
        """
        if not conf.d2rl_slightlycritical and not conf.precise_criticality_flag:
            return False
        elif self._recent_critical is None or self.vehicle.simulator.get_time() != self._recent_critical["time_step"]:
            self._recent_critical = self.update_critical()
        return self._recent_critical["slightly_critical"]

    # @profile
    def update_critical(self):
        """Update the criticality of the CAV.

        Returns:
            dict: Updated criticality of the CAV.
        """
        updated_critical = {
            "time_step": self.vehicle.simulator.get_time(),
            "highly_critical": False,
            "slightly_critical": False,
            "criticality": 0.0
            }
        criticality = 0.0
        if conf.env_mode == "NADE":
            if self.vehicle.simulator.env.eval_flag:
                criticality = conf.precise_criticality_threshold
            else:
                if not conf.simulation_config["neuralmetric_flag"]:
                # model-based criticality
                    for global_controller in self.vehicle.simulator.env.global_controller_instance_list:
                        if global_controller.veh_type == "BV":
                            criticality = np.sum(global_controller.original_NADE_decision.bv_criticality_list)
                            break
                else:
                # neural metric criticality
                    criticality = self.NN_metric.inference(self.cav_neuralmetric_obs_space)
            if not conf.precise_criticality_flag:
                if criticality > 0:
                    updated_critical["highly_critical"] = True
                elif conf.d2rl_slightlycritical:
                    obs = self.cav_obs_space
                    _, updated_critical["slightly_critical"] = RLControllerNew.calculate_cav_criticality(obs)
            else:
                if conf.precise_criticality_threshold > 0:
                    if criticality >= conf.precise_criticality_threshold:
                        updated_critical["highly_critical"] = True
                    elif criticality > 0:
                        updated_critical["slightly_critical"] = True
                else:
                    if criticality > 0:
                        updated_critical["highly_critical"] = True
        updated_critical["criticality"] = criticality
        return updated_critical

    def helper_solve_for_control_inputs(self, action, acc_dict):
        """Solve for control inputs.

        Args:
            action (list): Action list.
            acc_dict (dict): Acceleration dictionary.

        Returns:
            list: Solved control inputs.
        """
        acc_x, acc_y = acc_dict["acc_x"], acc_dict["acc_y"]
        guess_action = {"acceleration": action[0], "steering_angle": action[1]}
        new_states = update_vehicle_real_states(self.vehicle.states, guess_action, self.vehicle.params, self.vehicle.step_size)
        dx = new_states[1] - self.vehicle.states[1]
        current_v = self.vehicle.observation.information["Ego"]["velocity"]
        new_v = dx/self.vehicle.step_size*2-current_v
        acc = (new_v-current_v)/self.vehicle.step_size
        dy = new_states[2] - self.vehicle.states[2]
        new_lateral_v = dy/self.vehicle.step_size*2-self.vehicle.lateral_speed
        acc_lat =(new_lateral_v-self.vehicle.lateral_speed)/self.vehicle.step_size
        return [acc_x-acc, acc_y-acc_lat]

    @staticmethod
    # @profile
    def calculate_cav_criticality(train_state, prediction_time=2.0):
        """Calculate the criticality of the CAV.

        Args:
            train_state (dict): Training state of the CAV.
            prediction_time (float, optional): Prediction time. Defaults to 2.0.

        Returns:
            float: Criticality of the CAV.
            bool: Slightly critical flag of the CAV.
        """
        criticality, slight_flag = 0, False
        CAV_context = NDE._transfer_state_to_obs(train_state)
        considered_bv_list = RLControllerNew.find_considered_bv_list(CAV_context)
        all_obs = {}
        for veh_id in CAV_context.keys():
            all_obs[veh_id] = RLControllerNew.plain_process_information(veh_id, CAV_context)
        # CAV_obs = all_obs["CAV"]
        # full_obs = RLControllerNew.get_full_obs_from_cav_obs_and_bv_list(CAV_obs, CAV_context, considered_bv_list)
        # bv_critical_list = RLControllerNew.help_criticality_list(CAV_obs, CAV_context, full_obs, considered_bv_list, all_obs)
        # criticality = np.sum(bv_critical_list)
        if criticality == 0:
            CAV_context_new = RLControllerNew.helper_predict_new_cav_context(CAV_context, all_obs, prediction_time-1.0)
            all_obs_new = {}
            for veh_id in CAV_context_new.keys():
                all_obs_new[veh_id] = RLControllerNew.plain_process_information(veh_id, CAV_context_new)
            CAV_obs_new = all_obs_new["CAV"]
            considered_bv_list_new = RLControllerNew.find_considered_bv_list(CAV_context_new)
            full_obs_new = RLControllerNew.get_full_obs_from_cav_obs_and_bv_list(CAV_obs_new, CAV_context_new, considered_bv_list_new)
            bv_critical_list_new = RLControllerNew.help_criticality_list(CAV_obs_new, CAV_context_new, full_obs_new, considered_bv_list_new, all_obs_new)
            criticality_new = np.sum(bv_critical_list_new)
            if criticality_new > 0:
                criticality = criticality_new
                slight_flag = True
                # CAV_context_new = RLControllerNew.helper_predict_new_cav_context(CAV_context, all_obs, prediction_time-1.0)
                # all_obs_new = {}
                # for veh_id in CAV_context_new.keys():
                #     all_obs_new[veh_id] = RLControllerNew.plain_process_information(veh_id, CAV_context_new)
                # CAV_obs_new = all_obs_new["CAV"]
                # considered_bv_list_new = RLControllerNew.find_considered_bv_list(CAV_context_new)
                # full_obs_new = RLControllerNew.get_full_obs_from_cav_obs_and_bv_list(CAV_obs_new, CAV_context_new, considered_bv_list_new)
                # bv_critical_list_new = RLControllerNew.help_criticality_list(CAV_obs_new, CAV_context_new, full_obs_new, considered_bv_list_new, all_obs_new)
        return criticality, slight_flag
    
    @staticmethod
    # @profile
    def plain_process_information(ego_id, cav_context):
        """Process the information of the CAV.

        Args:
            ego_id (str): Ego vehicle ID.
            cav_context (dict): CAV context information.

        Returns:
            dict: Processed information of the CAV.
        """
        information = {}
        information["Ego"] = RLControllerNew.helper_plain_process_information(ego_id, ego_id, cav_context)
        ego_lane_index = information["Ego"]["lane_index"]
        leader, follower = RLControllerNew.helper_find_leader_follower(ego_id, cav_context, ego_lane_index, 0)
        leftleader, leftfollower = RLControllerNew.helper_find_leader_follower(ego_id, cav_context, ego_lane_index, 1)
        rightleader, rightfollower = RLControllerNew.helper_find_leader_follower(ego_id, cav_context, ego_lane_index, -1)
        information["Lead"] = RLControllerNew.helper_plain_process_information(leader, ego_id, cav_context)
        information["Foll"] = RLControllerNew.helper_plain_process_information(follower, ego_id, cav_context)
        information["LeftLead"] = RLControllerNew.helper_plain_process_information(leftleader, ego_id, cav_context)
        information["LeftFoll"] = RLControllerNew.helper_plain_process_information(leftfollower, ego_id, cav_context)
        information["RightLead"] = RLControllerNew.helper_plain_process_information(rightleader, ego_id, cav_context)
        information["RightFoll"] = RLControllerNew.helper_plain_process_information(rightfollower, ego_id, cav_context)
        return information

    @staticmethod
    # @profile
    def helper_plain_process_information(veh_id, ego_id, cav_context):
        """Helper function to process the information of the vehicle.

        Args:
            veh_id (str): Vehicle ID.
            ego_id (str): Ego vehicle ID.
            cav_context (dict): CAV context information.

        Returns:
            dict: Processed information of the vehicle.
        """
        if veh_id == None:
            return None

        veh = {"veh_id": veh_id}
        if veh_id == ego_id:
            distance = 0
        else:
            if cav_context[veh_id][66][0] > cav_context[ego_id][66][0]:
                distance = cav_context[veh_id][66][0]-cav_context[ego_id][66][0]-RLControllerNew.veh_length
            else:
                distance = cav_context[ego_id][66][0]-cav_context[veh_id][66][0]-RLControllerNew.veh_length
        lane_index = 0
        for i in range(len(RLControllerNew.road_geo)):
            road_bound = RLControllerNew.road_geo[i]
            if cav_context[veh_id][66][1] >= road_bound[0] and cav_context[veh_id][66][1] <= road_bound[1]:
                lane_index = i
                break
        veh["could_drive_adjacent_lane_left"] = (lane_index < 2)
        veh["could_drive_adjacent_lane_right"] = (lane_index > 0)
        veh["distance"] = distance
        abs_heading = np.arctan(abs(cav_context[veh_id][50]/cav_context[veh_id][64]))
        veh["heading"] = 90-np.sign(cav_context[veh_id][50])*np.degrees(abs_heading)
        veh["lane_index"] = lane_index
        veh["lateral_speed"] = cav_context[veh_id][50]
        veh["lateral_offset"] = cav_context[veh_id][66][1]-RLControllerNew.road_center[lane_index]
        veh["prev_action"] = None
        veh["position"] = cav_context[veh_id][66]
        veh["position3D"] = None
        veh["velocity"] = cav_context[veh_id][64]
        veh["road_id"] = None
        veh["acceleration"] = 0.
        return veh

    @staticmethod
    # @profile
    def helper_find_leader_follower(ego_id, cav_context, ego_lane_index, relative_lane_index):
        """Helper function to find the leader and follower of the vehicle.

        Args:
            ego_id (str): Ego vehicle ID.
            cav_context (dict): CAV context information.
            ego_lane_index (int): Ego lane index.
            relative_lane_index (int): Relative lane index.

        Returns:
            str: Leader vehicle ID.
            str: Follower vehicle ID.
        """
        ego_pos = cav_context[ego_id][66]
        leader, follower = None, None
        for veh_id in set(cav_context.keys())-{ego_id}:
            bv_lane_index = 0
            for i in range(len(RLControllerNew.road_geo)):
                road_bound = RLControllerNew.road_geo[i]
                if cav_context[veh_id][66][1] >= road_bound[0] and cav_context[veh_id][66][1] <= road_bound[1]:
                    bv_lane_index = i
                    break
            if bv_lane_index == ego_lane_index+relative_lane_index:
                if ego_pos[0] < cav_context[veh_id][66][0]: # leader
                    if not leader:
                        leader = veh_id
                    elif cav_context[veh_id][66][0] < cav_context[leader][66][0]:
                        leader = veh_id
                else: # follower
                    if not follower:
                        follower = veh_id
                    elif cav_context[veh_id][66][0] > cav_context[follower][66][0]:
                        follower = veh_id
        return leader, follower

    @staticmethod
    # @profile
    def find_considered_bv_list(cav_context):
        """Find the considered BV list.

        Args:
            cav_context (dict): CAV context information.

        Returns:
            list: Considered BV list.
        """
        cav_info = cav_context["CAV"]
        cav_pos = cav_info[66]
        bv_candidates = []
        bv_whole_list = []
        for id in cav_context.keys():
            if id != "CAV":
                bv_pos = cav_context[id][66]
                dist = utils.cal_euclidean_dist(cav_pos, bv_pos)
                if dist <= conf.cav_obs_range:
                    bv_whole_list.append([id, dist])
        bv_whole_list.sort(key=lambda i: i[1])
        for i in range(len(bv_whole_list)):
            if i < RLControllerNew.considered_bv_num:
                bv_id = bv_whole_list[i][0]
                bv_candidates.append(bv_id)
        return bv_candidates

    @staticmethod
    # @profile
    def get_full_obs_from_cav_obs_and_bv_list(CAV_obs, CAV_context, considered_bv_list):
        """Get the full observation from the CAV observation and BV list.

        Args:
            CAV_obs (dict): CAV observation.
            CAV_context (dict): CAV context information.
            considered_bv_list (list): Considered BV list.

        Returns:
            dict: Full observation.
        """
        # This observation will be a dict containing CAV and all BV NADE candidates
        full_obs = collections.OrderedDict()
        full_obs["CAV"] = CAV_obs["Ego"]
        for veh_id in considered_bv_list:
            full_obs[veh_id] = RLControllerNew.helper_plain_process_information(veh_id, "CAV", CAV_context)
        return full_obs

    @staticmethod
    # @profile
    def help_criticality_list(CAV_obs, CAV_context, full_obs, considered_bv_list, BVs_obs):
        """Helper function to calculate the criticality list.

        Args:
            CAV_obs (dict): CAV observation.
            CAV_context (dict): CAV context information.
            full_obs (dict): Full observation.
            considered_bv_list (list): Considered BV list.
            BVs_obs (dict): BVs observation.

        Returns:
            list: Criticality list.
        """
        bv_criticality_list, criticality_array_list = [], []
        predicted_full_obs, predicted_traj_obs = NADEBVGlobalController.pre_load_predicted_obs_and_traj(full_obs)
        # predicted_full_obs, predicted_traj_obs = None, None
        CAV_left_prob, CAV_still_prob, CAV_right_prob = NADEBVGlobalController._get_Surrogate_CAV_action_probability(cav_obs=CAV_obs)
        for bv_id in considered_bv_list:
            BV_obs = BVs_obs[bv_id]
            bv_criticality, criticality_array = RLControllerNew.helper_decompose_decision(
                BV_obs, CAV_obs, SM_LC_prob=[CAV_left_prob, CAV_still_prob, CAV_right_prob], full_obs=full_obs, predicted_full_obs=predicted_full_obs, predicted_traj_obs=predicted_traj_obs)
            bv_criticality_list.append(bv_criticality), criticality_array_list.append(criticality_array)
        return bv_criticality_list

    @staticmethod
    # @profile
    def helper_decompose_decision(BV_obs, CAV, SM_LC_prob, full_obs=None, predicted_full_obs=None, predicted_traj_obs=None):
        """Helper function to decompose the decision.

        Args:
            BV_obs (dict): BV observation.
            CAV (dict): CAV observation.
            SM_LC_prob (list): Surrogate model lane change probability.
            full_obs (dict, optional): Full observation. Defaults to None.
            predicted_full_obs (dict, optional): Predicted full observation. Defaults to None.
            predicted_traj_obs (dict, optional): Predicted trajectory observation. Defaults to None.

        Returns:
            float: BV criticality.
            list: Criticality array.
        """
        bv_criticality, criticality_array = -np.inf, np.zeros(len(conf.ACTIONS), dtype=float)
        _,_,bv_pdf = NDDController.static_get_ndd_pdf(obs=BV_obs)
        bv_left_prob, bv_right_prob = bv_pdf[0], bv_pdf[1]
        # If in lane change mode or lane change prob = 1, then not control!
        if not ((0.99999 <= bv_left_prob <= 1) or (0.99999 <= bv_right_prob <= 1)):
            bv_criticality, criticality_array,_,_ = TreeSearchNADEBackgroundController._calculate_criticality(
                BV_obs, CAV, SM_LC_prob, full_obs, predicted_full_obs, predicted_traj_obs)
        return bv_criticality, criticality_array

    @staticmethod
    # @profile
    def helper_predict_new_cav_context(CAV_context, all_obs, prediction_time):
        """Helper function to predict the new CAV context.

        Args:
            CAV_context (dict): CAV context information.
            all_obs (dict): All observations.
            prediction_time (float): Prediction time.

        Returns:
            dict: New CAV context information.
        """
        CAV_context_new = {}
        for veh_id in CAV_context.keys():
            lat_speed = CAV_context[veh_id][50]
            long_speed = CAV_context[veh_id][64]
            current_x, current_y = CAV_context[veh_id][66]
            lane_index = all_obs[veh_id]["Ego"]["lane_index"]
            y_offset = all_obs[veh_id]["Ego"]["lateral_offset"]
            acc = acceleration(all_obs[veh_id]["Ego"], all_obs[veh_id]["Lead"], veh_id)
            new_x = current_x + long_speed*prediction_time + 0.5*acc*(prediction_time**2)
            if veh_id != "CAV": # BV's lane change must be completed
                if lat_speed > 0:
                    if y_offset >= 0:
                        new_lane_index = min(lane_index+1, 2)
                    else:
                        new_lane_index = lane_index
                elif lat_speed < 0:
                    if y_offset <= 0:
                        new_lane_index = max(lane_index-1, 0)
                    else:
                        new_lane_index = lane_index
                else:
                    new_lane_index = lane_index
                new_y = RLControllerNew.road_center[new_lane_index]
                new_lat_speed = 0
            else:
                new_y_tmp = current_y + lat_speed*prediction_time
                new_y = np.clip(new_y_tmp, RLControllerNew.road_center[0], RLControllerNew.road_center[-1])
                if new_y_tmp != new_y:
                    new_lat_speed = 0
                else:
                    new_lat_speed = lat_speed
            CAV_context_new[veh_id] = {
                50: new_lat_speed,
                64: long_speed+acc*prediction_time,
                66: (new_x, new_y)
            }
        return CAV_context_new