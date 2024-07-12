from numpy import random
from numpy.core.numeric import full
from controller.treesearchnadecontroller import TreeSearchNADEBackgroundController
import numpy as np
from copy import deepcopy
import collections
import utils
import os
if os.environ["mode"] == "testing":
    import conf.conf_testing as conf
elif os.environ["mode"] == "training":
    import conf.conf_training as conf
else:
    raise ValueError("Please set the mode to testing or training")
from controller.nddglobalcontroller import NDDBVGlobalController
from controller.nddcontroller import NDDController
from controller.nadecontroller import NADEBackgroundController
# from controller.treesearchnadecontroller_original import TreeSearchNADEBackgroundController
drl_action_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,0.995,0.999,1.0-1e-9]

class decision_information:
    def __init__(self, sumo_time_stamp, bv_criticality_list, criticality_array_list, bv_action_idx_list, weight_list, ndd_possi_list, IS_possi_list):
        """Initialize the decision_information class.

        Args:
            sumo_time_stamp (float): SUMO time stamp.
            bv_criticality_list (list): Total criticality of each BV.
            criticality_array_list (list(array)): Criticality list of each BV maneuver.
            bv_action_idx_list (list): Index of the selected BV action.
            weight_list (list): Weight of the each BV action.
            ndd_possi_list (list): NDD pdf of each BV action.
            IS_possi_list (list): Importance sampling pdf of each BV action.
        """
        self.sumo_time_stamp = sumo_time_stamp
        self.bv_criticality_list = bv_criticality_list
        self.criticality_array_list = criticality_array_list
        self.bv_action_idx_list = bv_action_idx_list
        self.weight_list = weight_list
        self.ndd_possi_list = ndd_possi_list
        self.IS_possi_list = IS_possi_list
        self.predicted_full_obs, self.predicted_traj_obs = None, None


class NADEBVGlobalController(NDDBVGlobalController):
    controlled_bv_num = 4

    def __init__(self, env, veh_type="BV"):
        """Initialize the NADEBVGlobalController class.

        Args:
            env (NADE): The NADE environment.
            veh_type (str, optional): The type of the vehicle. Defaults to "BV".
        """
        super().__init__(env, veh_type)
        self.vehicle_history_traj = {}
        # self.discriminator_agent = conf.discriminator_agent
        self.drl_info = None
        self.drl_epsilon_value = -1
        self.real_epsilon_value = -1
        self._recent_original_NADE_decision = None
        self.original_NADE_controlled_bvs_ID_set = set()
        self.nade_info = {}
        self.control_log = {"criticality":0, "discriminator_input":0, "weight_list_per_simulation":[1]}
        self.nade_candidates = []
        self.predicted_full_obs = None
        self.predicted_traj_obs = None

    def apply_control_permission(self):
        """Check if the controlled BVs can be controlled. When the BV is in the lane change process, it cannot be controlled.

        Returns:
            bool: True if the controlled BVs can be controlled, False otherwise.
        """
        for vehicle in self.get_bv_candidates():
            if vehicle.controller.NADE_flag and utils.is_lane_change(vehicle.observation.information["Ego"]):
                return False
        if conf.simulation_config["map"] == "Mcity":
            cav_ego_obs = self.env.vehicle_list["CAV"].observation.information["Ego"]
            if not(cav_ego_obs["road_id"] == "EG_1_3_1" or cav_ego_obs["road_id"] == "EG_1_1_1"):
                return False
        return True

    # @profile
    def project_position_from_ACM_to_straight_lane(self, vehicle):
        """Project position from curved lanes to straight lanes.

        Args:
            vehicle (Vehicle): Vehicle instance of the studied vehicle.

        Raises:
            ValueError: Vehicle lane change information is wrong.

        Returns:
            float: Projected x position.
            float: Projected y position.
        """
        initial_edge_id = '-1006000.250.21'
        
        initial_position = 0
        # vehicle_edge_id = self.env.simulator.get_road_ID(vehicle.id)
        # # vehicle_edge_info = self.env.route_edges_info[vehicle_edge_id]
        # vehicle_lane_id = self.env.simulator.get_vehicle_laneID(vehicle.id)
        vehicle_distance = self.env.simulator.get_vehicle_distance_to_edge(vehicle.id, initial_edge_id, initial_position) # projected x
        could_change_left = self.env.simulator.get_vehicle_lane_adjacent(vehicle.id,1)
        could_change_right = self.env.simulator.get_vehicle_lane_adjacent(vehicle.id,-1)
        if could_change_left and not could_change_right:
            vehicle_relative_lane_index_to_ini = 0
        elif could_change_right and not could_change_left:
            vehicle_relative_lane_index_to_ini = 1
        else:
            raise ValueError("wrong bv lane change info")
        vehicle_offset = self.env.simulator.get_vehicle_lateral_lane_position(vehicle.id)
        vehicle_lane_width = self.env.simulator.get_vehicle_lane_width(vehicle.id)
        new_x = vehicle_distance
        new_y = conf.lane_list[vehicle_relative_lane_index_to_ini] + vehicle_offset*4/vehicle_lane_width
        return new_x, new_y
    
    def update_history_traj(self):
        """Update the history trajectory of the vehicles.
        """
        time = self.env.simulator.get_time()
        real_veh_list = self.env.simulator.get_vehID_list()
        for vehicle in self.env.vehicle_list:
            if vehicle.id in real_veh_list:
                if vehicle.id not in self.vehicle_history_traj:
                    self.vehicle_history_traj[vehicle.id] = {} 
                if conf.simulation_config["map"] != "Mcity" and conf.simulation_config["map"] != "ACM":
                    self.vehicle_history_traj[vehicle.id][str(time)] = vehicle.observation.information["Ego"]["position"]
                else:
                    self.vehicle_history_traj[vehicle.id][str(time)] = self.project_position_from_ACM_to_straight_lane(vehicle)
        
        delete_vehicle_key = []    
        for veh_id in self.vehicle_history_traj:
            if veh_id not in self.env.vehicle_list:
                delete_vehicle_key.append(veh_id)
            else:
                delete_key_tmp = []
                for key in self.vehicle_history_traj[veh_id]:
                    if float(key) < time-2:
                        delete_key_tmp.append(key)
                for key in delete_key_tmp:
                    self.vehicle_history_traj[veh_id].pop(key)
        for veh_id in delete_vehicle_key:
            self.vehicle_history_traj.pop(veh_id)

    # @profile
    def step(self, drl_action=None):
        """Control the selected vehicle to realize the decided behavior.

        Args:
            drl_action (int, optional): The action index from the DRL agent. Defaults to None.

        Returns:
            list: List of the vehicle criticality.
        """
        self.real_epsilon_value = -1
        self.drl_epsilon_value = -1
        if drl_action is None:
            drl_action = self.env.drl_action
        if drl_action is not None:
            # drl_action = drl_action/10.0 # ! delete for PPO, should be enable when DQN is used
            if conf.simulation_config["epsilon_type"] == "discrete":
                drl_action = drl_action_list[drl_action] # discrete version
            if drl_action > 1:
                drl_action = 1
            elif drl_action < 0:
                drl_action = 0
        self.update_history_traj()
        self.control_log = {"criticality":0, "discriminator_input":0}
        snap_nade_info = {}
        bv_action_idx_list, weight_list, max_vehicle_criticality, ndd_possi_list, IS_possi_list, controlled_bvs_list = [], [], [], [], [], []
        vehicle_criticality_list = []
        self.reset_control_and_action_state()
        self.update_subscription(controller=TreeSearchNADEBackgroundController)
        for bv_id in self.controllable_veh_id_list:
            bv = self.env.vehicle_list[bv_id]
            if self.env.simulator.replay_flag or conf.train_mode == "offline":
                bv.update()
            bv.controller.step()
        # if self.apply_control_permission():
        if not self.env.eval_flag:
            bv_action_idx_list, weight_list, max_vehicle_criticality, ndd_possi_list, IS_possi_list, controlled_bvs_list, vehicle_criticality_list, _, bv_action_criticality_list = self.select_controlled_bv_and_action(drl_action)
            # self.drl_info["prev_action"] = None
            for bv_id in self.controllable_veh_id_list:
                bv = self.env.vehicle_list[bv_id]
                # bv.controller.step()
                if self.apply_control_permission():
                    if bv in controlled_bvs_list:
                        nade_action = bv_action_idx_list[controlled_bvs_list.index(bv)]
                        nade_action_criticality = bv_action_criticality_list[controlled_bvs_list.index(bv)]
                        if nade_action is not None:
                            # new_discriminator_action = self.vehicle_action_to_discriminator_action(bv, controlled_bvs_list, nade_action)
                            # self.drl_info["prev_action"] = new_discriminator_action
                            if conf.experiment_config["mode"] != "risk_NDE":
                                self.control_log["ndd_possi"] = ndd_possi_list[controlled_bvs_list.index(bv)]
                                snap_nade_info[bv.id] = [nade_action,nade_action_criticality]
                                bv.controller.action = utils.action_id_to_action_command(
                                    nade_action)
                                bv.controller.NADE_flag = True
                                bv.simulator.set_vehicle_color(
                                    bv.id, bv.color_blue)
                                # print(bv.id, nade_action) 
                    if not self.env.simulator.replay_flag and conf.train_mode != "offline":
                        bv.update()
        # if self.drl_info["prev_action"] is None:
        #     self.drl_info["prev_action"] = 0        
        self.control_log["weight_list_per_simulation"] = [
            val for val in weight_list if val is not None]
        if len(self.control_log["weight_list_per_simulation"]) == 0:
            self.control_log["weight_list_per_simulation"] = [1]
        if snap_nade_info != {}:
            self.nade_info[str(self.env.simulator.sumo_time_stamp)] = snap_nade_info
        elif conf.debug_critical:
            self.nade_info[str(self.env.simulator.sumo_time_stamp)] = self.output_criticality_info_debug()
        return vehicle_criticality_list

    def output_criticality_info_debug(self):
        """Output the criticality information for debugging.

        Returns:
            dict: Dictionary of the criticality information.
        """
        debug_info = {
            "bv_criticality_list": self.original_NADE_decision.bv_criticality_list
        }
        pred_info = {}
        for veh_id in self.predicted_full_obs.keys():
            if veh_id == "CAV":
                pred_info[veh_id] = {"SM_LC_PROB": self.SM_LC_prob}
            else:
                pred_info[veh_id] = {"challenge": list(self.env.vehicle_list[veh_id].controller.bv_challenge_array)}
            for action in self.predicted_full_obs[veh_id].keys():
                ind_pred_info = self.predicted_full_obs[veh_id][action]
                pred_info[veh_id]["cur_pos"] = ind_pred_info["position3D"]
                pred_info[veh_id][action] = ind_pred_info["position"]
        debug_info["pred"] = pred_info
        return debug_info

    def vehicle_action_to_discriminator_action(self, bv, controlled_bv_list, action):
        """Convert the vehicle action to the discriminator action.

        Args:
            bv (Vehicle): The vehicle instance.
            controlled_bv_list (list): List of controlled BVs.
            action (int): The vehicle action.

        Returns:
            int: The discriminator action.
        """
        bv_index = controlled_bv_list.index(bv)
        if action < 2:
            action_number = action
        else:
            action_number = 2 
        discriminator_action = bv_index * 3 + action_number + 1
        return discriminator_action


    @staticmethod
    # @profile
    def get_reachability_range(current_pos, current_velocity, predict_time_gap=1, max_a=2, min_a=-4):
        """Return the predicted reachability range of each vehicle.

        Args:
            current_pos (float): Current position.
            current_velocity (float): Current velocity.
            predict_time_gap (float): The desired predicted time.
            max_a (float): Maximal acceleraton.
            min_a (float): Minimal acceleration.

        Returns:
            tuple: Upper bound and lower bound of the vehicle's reachability range.
        """
        lower_bound_list = []
        upper_bound_list = []
        time_interval_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
        # time_interval_list = [1]
        lower_bound_list = [current_pos + current_velocity * predict_time_gap + 0.5*min_a*predict_time_gap**2 for predict_time_gap in time_interval_list]
        upper_bound_list = [current_pos + current_velocity * predict_time_gap + 0.5*max_a*predict_time_gap**2 for predict_time_gap in time_interval_list]
        # lower_bound = current_pos + current_velocity * predict_time_gap + 0.5*min_a*predict_time_gap**2
        # upper_bound = current_pos + current_velocity * predict_time_gap + 0.5*max_a*predict_time_gap**2
        return lower_bound_list, upper_bound_list

    @staticmethod
    # @profile
    def intersection_check(cav_reach_range_lb_list, cav_reach_range_ub_list, bv_reach_range_lb_list, bv_reach_range_ub_list):
        """Return whether the reachability ranges for CAV and BV have intersections.

        Args:
            cav_reach_range_lb_list (list): List of lower bound of the CAV reachability range.
            cav_reach_range_ub_list (list): List of upper bound of the CAV reachability range.
            bv_reach_range_lb_list (list): List of lower bound of the BV reachability range.
            bv_reach_range_ub_list (list): List of upper bound of the BV reachability range.

        Returns:
            bool: True means the reachability ranges have intersection, False means no intersection.
        """
        intersection_flag = False
        for i in range(len(cav_reach_range_lb_list)):
            cav_lb, cav_ub, bv_lb, bv_ub = cav_reach_range_lb_list[i], cav_reach_range_ub_list[i], bv_reach_range_lb_list[i], bv_reach_range_ub_list[i]
            tmp_intersection_flag = (max(cav_lb, bv_lb) <= min(cav_ub, bv_ub))
            intersection_flag = intersection_flag and tmp_intersection_flag
        return intersection_flag

    # @profile
    def get_bv_candidates(self):
        """Find the BV candidates around the CAV to take the challenging maneuvers.

        Returns:
            list(Vehicle): List of background vehicles around the CAV.
        """        
        av = self.env.vehicle_list["CAV"]
        av_pos = av.observation.local[av.id][66]
        av_velocity = av.observation.local[av.id][64]
        cav_reach_range_lb_list, cav_reach_range_ub_list = NADEBVGlobalController.get_reachability_range(av_pos[0], av_velocity)
        av_contex = av.observation.context
        bv_ID_list = list(av_contex.keys())
        bv_candidates = []
        bv_list = []
        for id in bv_ID_list:
            bv_pos = av_contex[id][66]
            dist = utils.cal_euclidean_dist(av_pos, bv_pos)
            if dist <= conf.cav_obs_range:
                bv_list.append([id, dist])
        bv_list.sort(key=lambda i: i[1])
        realtime_vehicle_list = self.env.simulator.get_vehID_list()
        for i in range(len(bv_list)):
            if i < self.controlled_bv_num:
                bv_id = bv_list[i][0]
                if bv_id in self.env.vehicle_list.keys() and bv_id in realtime_vehicle_list:
                    bv = self.env.vehicle_list[bv_id]
                    if conf.simulation_config["map"] == "Mcity":
                        if not (bv.observation.information["Ego"]["road_id"] == "EG_1_3_1" or bv.observation.information["Ego"]["road_id"] == "EG_1_1_1"):
                            continue
                    # if self.env.simulator.get_vehicle_type_id(id) == "TRUCK":
                    #     continue
                    # bv_dist = bv_list[i][1]
                    bv_candidates.append(bv)
                    # bv_id = bv_list[i][0]
                    # bv = self.env.vehicle_list[bv_id]
                    # bv_pos = bv.observation.information["Ego"]["position"][0]
                    # bv_velocity = bv.observation.information["Ego"]["velocity"]
                    # bv_reach_range_lb_list, bv_reach_range_ub_list = NADEBVGlobalController.get_reachability_range(bv_pos, bv_velocity)
                    # if NADEBVGlobalController.intersection_check(cav_reach_range_lb_list, cav_reach_range_ub_list, bv_reach_range_lb_list, bv_reach_range_ub_list):
                    #     # cav and bv has overlap
                    #     bv_candidates.append(bv)
        return bv_candidates

    @staticmethod
    # @profile
    def pre_load_predicted_obs_and_traj(full_obs):
        """Reload the predicted observation and trajectory observation.
        
        Args:
            full_obs (dict): Full observation of the vehicles.
            
        Returns:
            dict: Predicted observation of the vehicles.
            dict: Trajectory of the vehicles.
        """
        predicted_obs = {}
        trajectory_obs = {}
        action_list_bv = ["left", "right", "still", "brake", "constant"]
        action_list_cav = ["left", "right", "still", "brake", "constant", "accelerate"]
        for veh_id in full_obs:
            if veh_id == "CAV":
                action_list = action_list_cav
            else:
                action_list = action_list_bv
            for action in action_list:
                vehicle = full_obs[veh_id]
                if veh_id not in trajectory_obs:
                    trajectory_obs[veh_id] = {}
                if veh_id not in predicted_obs:
                    predicted_obs[veh_id] = {}
                predicted_obs[veh_id][action], trajectory_obs[veh_id][action] = TreeSearchNADEBackgroundController.update_single_vehicle_obs(vehicle, action)
        return predicted_obs, trajectory_obs
    

    @staticmethod
    def change_criticality_result(bv_criticality_list, controlled_bv_num):
        """Change the criticality result to the desired format.

        Args:
            bv_criticality_list (list): List of the criticality of the BVs.
            controlled_bv_num (int): Number of controlled BVs.
            
        Returns:
            list: Processed criticality of the BVs.
        """
        for i in range(len(bv_criticality_list)):
            if bv_criticality_list[i] > 0:
                bv_criticality_list[i] = max(-20, np.log10(bv_criticality_list[i]))
            else:
                bv_criticality_list[i] = -20
        if len(bv_criticality_list) < controlled_bv_num:
            bv_criticality_list.extend([-20]*(controlled_bv_num-len(bv_criticality_list)))
        return bv_criticality_list
    
    def collect_discriminator_input_simplified(self, full_obs, controlled_bvs_list, bv_criticality_list):
        """Collect the input for the NADE agent.

        Args:
            full_obs (dict): Full observation of the vehicles.
            controlled_bvs_list (list): List of controlled BVs.
            bv_criticality_list (list): List of the criticality of the BVs.

        Returns:
            np.array: Array of the input for the NADE agent.
        """
        CAV_global_position = list(full_obs["CAV"]["position"])
        CAV_speed = full_obs["CAV"]["velocity"]
        CAV_history_traj = full_obs["CAV"]["history"]
        tmp_weight = self.env.info_extractor.episode_log["weight_episode"]
        # tmp_weight = max(np.log(tmp_weight), -40)
        tmp_weight = np.log10(tmp_weight)
        vehicle_info_list = []
        controlled_bv_num = 1
        total_bv_info_length = controlled_bv_num * 4
        # print(bv_criticality_list)
        if len(bv_criticality_list):
            selected_bv_index = np.argmax(np.array(bv_criticality_list))
            vehicle = controlled_bvs_list[selected_bv_index]
            veh_id = vehicle.id
            vehicle_single_obs = full_obs[veh_id]
            vehicle_local_position = list(vehicle_single_obs["position"])
            vehicle_relative_position = [vehicle_local_position[0]-CAV_global_position[0], vehicle_local_position[1]-CAV_global_position[1]]
            vehicle_relative_speed = vehicle_single_obs["velocity"] - CAV_speed
            predict_relative_position = vehicle_relative_position[0] + vehicle_relative_speed
            vehicle_info_list.extend(vehicle_relative_position +[vehicle_relative_speed] + [predict_relative_position])
        else:
            vehicle_info_list.extend([-20, -8, -10, -20])

        if len(vehicle_info_list) < total_bv_info_length:
            vehicle_info_list.extend([-1]*(total_bv_info_length - len(vehicle_info_list)))
        bv_criticality_flag = (sum(bv_criticality_list) > 0)
        if sum(bv_criticality_list) > 0:
            bv_criticality_value = np.log10(sum(bv_criticality_list))
        else:
            bv_criticality_value = 16
        if conf.simulation_config["map"] == "2LaneLong":
            CAV_position_lb, CAV_position_ub = [400, 40], [4400, 50]
        else:
            CAV_position_lb, CAV_position_ub = [400, 40], [800, 50]
        if conf.simulation_config["speed_mode"] == "high_speed":
            CAV_velocity_lb, CAV_velocity_ub = 20, 40
        else:
            CAV_velocity_lb, CAV_velocity_ub = 0, 20
        weight_lb = -30
        weight_ub = 0
        bv_criticality_flag_lb = 0
        bv_criticality_flag_ub = 1
        bv_criticality_value_lb = -16
        bv_criticality_value_ub = 0
        vehicle_info_lb, vehicle_info_ub = [-20, -8, -10, -20], [20, 8, 10, 20]
        lb_array = np.array(CAV_position_lb + [CAV_velocity_lb] + [weight_lb] + [bv_criticality_flag_lb] + [bv_criticality_value_lb] + vehicle_info_lb * controlled_bv_num)
        ub_array = np.array(CAV_position_ub + [CAV_velocity_ub] + [weight_ub] + [bv_criticality_flag_ub] + [bv_criticality_value_ub] + vehicle_info_ub * controlled_bv_num)

        total_obs_for_DRL_ori = np.array(CAV_global_position + [CAV_speed] + [tmp_weight] + [bv_criticality_flag] + [bv_criticality_value] + vehicle_info_list)
        total_obs_for_DRL = 2 * (total_obs_for_DRL_ori - lb_array)/(ub_array - lb_array) - 1
        total_obs_for_DRL = np.clip(total_obs_for_DRL, -5, 5)
        return np.float32(np.array(total_obs_for_DRL))

    def collect_discriminator_input_more_bvs(self, full_obs, controlled_bvs_list, bv_criticality_list): # 19 dimension observation
        """Collect the input for the NADE agent.
        
        Args:
            full_obs (dict): Full observation of the vehicles.
            controlled_bvs_list (list): List of controlled BVs.
            bv_criticality_list (list): List of the criticality of the BVs.

        Returns:
            np.array: Array of the input for the NADE agent.
        """
        CAV_global_position = list(full_obs["CAV"]["position"])
        CAV_speed = full_obs["CAV"]["velocity"]
        CAV_history_traj = full_obs["CAV"]["history"]
        tmp_weight = self.env.info_extractor.episode_log["weight_episode"]
        # tmp_weight = max(np.log(tmp_weight), -40)
        tmp_weight = np.log10(tmp_weight)
        vehicle_info_list = []
        controlled_bv_num = 3
        total_bv_info_length = 4 + controlled_bv_num * 3
        pov_id = None
        if len(bv_criticality_list):
            selected_bv_index = np.argmax(np.array(bv_criticality_list))
            vehicle = controlled_bvs_list[selected_bv_index]
            veh_id, pov_id = vehicle.id, vehicle.id
            vehicle_single_obs = full_obs[veh_id]
            vehicle_local_position = list(vehicle_single_obs["position"])
            vehicle_relative_position = [vehicle_local_position[0]-CAV_global_position[0], vehicle_local_position[1]-CAV_global_position[1]]
            vehicle_relative_speed = vehicle_single_obs["velocity"] - CAV_speed
            predict_relative_position = vehicle_relative_position[0] + vehicle_relative_speed
            vehicle_info_list.extend(vehicle_relative_position +[vehicle_relative_speed] + [predict_relative_position])
        else:
            vehicle_info_list.extend([-20, -8, -10, -20])
        
        # ! re-enable the information from multiple BVs
        for vehicle in controlled_bvs_list:
            veh_id = vehicle.id
            if veh_id != pov_id:
                vehicle_single_obs = full_obs[veh_id]
                vehicle_local_position = list(vehicle_single_obs["position"])
                vehicle_relative_position = [vehicle_local_position[0]-CAV_global_position[0], vehicle_local_position[1]-CAV_global_position[1]]
                vehicle_relative_speed = vehicle_single_obs["velocity"] - CAV_speed
                vehicle_info_list.extend(vehicle_relative_position +[vehicle_relative_speed])
        if len(vehicle_info_list) < total_bv_info_length:
            vehicle_info_list.extend([-20, -8, -10]*int((total_bv_info_length - len(vehicle_info_list))/3))
        bv_criticality_flag = (sum(bv_criticality_list) > 0)
        if sum(bv_criticality_list) > 0:
            bv_criticality_value = np.log10(sum(bv_criticality_list))
        else:
            bv_criticality_value = 16
        if conf.simulation_config["map"] == "2LaneLong":
            CAV_position_lb, CAV_position_ub = [400, 40], [4400, 50]
        else:
            CAV_position_lb, CAV_position_ub = [400, 40], [800, 50]
        CAV_velocity_lb, CAV_velocity_ub = 0, 20
        weight_lb = -30
        weight_ub = 0
        bv_criticality_flag_lb = 0
        bv_criticality_flag_ub = 1
        bv_criticality_value_lb = -16
        bv_criticality_value_ub = 0
        vehicle_info_lb, vehicle_info_ub = [-20, -8, -10, -20], [20, 8, 10, 20]
        lb_array = np.array(CAV_position_lb + [CAV_velocity_lb] + [weight_lb] + [bv_criticality_flag_lb] + [bv_criticality_value_lb] + vehicle_info_lb +  vehicle_info_lb[:-1]*(controlled_bv_num))
        ub_array = np.array(CAV_position_ub + [CAV_velocity_ub] + [weight_ub] + [bv_criticality_flag_ub] + [bv_criticality_value_ub] + vehicle_info_ub +  vehicle_info_ub[:-1]*(controlled_bv_num))
        
        total_obs_for_DRL_ori = np.array(CAV_global_position + [CAV_speed] + [tmp_weight] + [bv_criticality_flag] + [bv_criticality_value] + vehicle_info_list)
        total_obs_for_DRL = 2 * (total_obs_for_DRL_ori - lb_array)/(ub_array - lb_array) - 1
        total_obs_for_DRL = np.clip(total_obs_for_DRL, -5, 5)
        return np.array(total_obs_for_DRL)

    def _get_original_NADE_decision(self, controlled_bvs_list, CAV_obs, full_obs):
        """Helper function to calculate the NADE decision.

        Args:
            controlled_bvs_list (list): List of controlled BVs.
            CAV_obs (dict): Observation of the CAV.
            full_obs (dict): Full observation of the vehicles.

        Returns:
            decision_information: Decision information of the NADE.
        """
        bv_criticality_list, criticality_array_list, bv_action_idx_list, weight_list, ndd_possi_list, IS_possi_list = [], [], [], [], [], []        
        predicted_full_obs, predicted_traj_obs = NADEBVGlobalController.pre_load_predicted_obs_and_traj(full_obs)
        self.predicted_full_obs = predicted_full_obs
        self.predicted_traj_obs = predicted_traj_obs
        CAV_left_prob, CAV_still_prob, CAV_right_prob = NADEBVGlobalController._get_Surrogate_CAV_action_probability(cav_obs=self.env.vehicle_list["CAV"].observation.information)
        self.SM_LC_prob = [CAV_left_prob, CAV_still_prob, CAV_right_prob]
        for bv in controlled_bvs_list:
            try:
                bv_criticality, criticality_array = bv.controller.Decompose_decision(
                    CAV_obs, SM_LC_prob=[CAV_left_prob, CAV_still_prob, CAV_right_prob], full_obs=full_obs, predicted_full_obs=predicted_full_obs, predicted_traj_obs=predicted_traj_obs)
                bv_criticality_list.append(bv_criticality), criticality_array_list.append(criticality_array)
            except:
                bv_criticality_list.append(0.0), criticality_array_list.append(np.zeros(len(conf.ACTIONS), dtype=float))
        NADE_decision = decision_information(self.env.simulator.get_time(), bv_criticality_list, criticality_array_list, bv_action_idx_list, weight_list, ndd_possi_list, IS_possi_list)
        return NADE_decision

    @property
    def original_NADE_decision(self):
        """Get the NADE decision.

        Returns:
            decision_information: Decision information of the NADE agent.
        """
        controlled_bvs_list = self.get_bv_candidates()
        controlled_bvs_ID_set = set([v.id for v in controlled_bvs_list])
        if not self._recent_original_NADE_decision or self._recent_original_NADE_decision.sumo_time_stamp != self.env.simulator.get_time() or self.original_NADE_controlled_bvs_ID_set != controlled_bvs_ID_set: 
            #if the recent nade decision exists and the recent observation is not updated at the current timestamp, update the recent nade deicision
            self.original_NADE_controlled_bvs_ID_set = controlled_bvs_ID_set
            CAV_obs = self.env.vehicle_list["CAV"].observation.information
            full_obs = self.get_full_obs_from_cav_obs_and_bv_list(CAV_obs, controlled_bvs_list)
            self._recent_original_NADE_decision = self._get_original_NADE_decision(controlled_bvs_list, CAV_obs, full_obs)
        return self._recent_original_NADE_decision

    def get_underline_drl_action(self, discriminator_input, bv_criticality_list):
        """Get the underline NADE agent action.

        Args:
            discriminator_input (np.array): Array of the input for the NADE agent.
            bv_criticality_list (list): List of the criticality of the BVs.

        Returns:
            float: The underline NADE agent action.
        """
        underline_drl_action = None
        if sum(bv_criticality_list) > 0:
            if conf.simulation_config["epsilon_setting"] == "drl":
                try:
                    if sum(bv_criticality_list) < conf.criticality_threshold:
                        underline_drl_action = 1-1e-9 # if criticality less than 1e-6, return 0.99 as the epsilon
                    else:
                        if conf.nade_agent is None:
                            conf.nade_agent = conf.load_discriminator_agent()
                        underline_drl_action = conf.nade_agent.compute_action(discriminator_input)
                        if conf.simulation_config["epsilon_type"] == "discrete":
                            underline_drl_action = drl_action_list[underline_drl_action]
                        if underline_drl_action < 0:
                            underline_drl_action = 0
                        if underline_drl_action > 1:
                            underline_drl_action = 1
                except Exception as e:
                    print(e)
                    underline_drl_action = conf.epsilon_value
            elif conf.simulation_config["epsilon_setting"] == "fixed":
                underline_drl_action = conf.epsilon_value
            elif conf.simulation_config["epsilon_setting"] == "varied":
                if sum(bv_criticality_list) < conf.criticality_threshold:
                    underline_drl_action = 1-1e-9
                else:
                    underline_drl_action = conf.epsilon_value
        return underline_drl_action

    # @profile
    def select_controlled_bv_and_action(self, drl_action):
        """Select the background vehicle controlled by NADE and the corresponding action.

        Returns:
            list(float): List of action index for all studied background vehicles. 
            list(float): List of weight of each vehicle.
            float: Maximum criticality.
            list(float): List of behavior probability based on NDD.
            list(float): List of critical possibility.
            list(Vehicle): List of all studied vehicles.
            list(float): List of vehicle criticality.
            np.array: Array of the input for the NADE agent.
            list(float): List of criticality of each vehicle.
        """
        num_controlled_critical_bvs = 2
        controlled_bvs_list = self.get_bv_candidates()
        CAV_obs = self.env.vehicle_list["CAV"].observation.information
        full_obs = self.get_full_obs_from_cav_obs_and_bv_list(CAV_obs, controlled_bvs_list)
        self.nade_candidates = controlled_bvs_list
        bv_criticality_list = self.original_NADE_decision.bv_criticality_list
        criticality_array_list = self.original_NADE_decision.criticality_array_list
        bv_action_idx_list = self.original_NADE_decision.bv_action_idx_list
        bv_action_criticality_list = []
        weight_list = self.original_NADE_decision.weight_list
        ndd_possi_list = self.original_NADE_decision.ndd_possi_list
        IS_possi_list = self.original_NADE_decision.IS_possi_list
        whole_weight_list = []
        self.control_log["criticality"] = sum(bv_criticality_list)
    
        discriminator_input = self.collect_discriminator_input_simplified(full_obs, controlled_bvs_list, bv_criticality_list)
        self.control_log["discriminator_input"] = discriminator_input.tolist()

        self.epsilon_value = -1
        underline_drl_action = self.get_underline_drl_action(discriminator_input, bv_criticality_list)
        
        if sum(bv_criticality_list) > 0:
            if drl_action is None:
                drl_action = underline_drl_action
                self.drl_epsilon_value = underline_drl_action
                self.real_epsilon_value = underline_drl_action
            else:
                # Decide whether the drl action will be replaced by the underlining drl action
                if conf.simulation_config["explore_mode"] == "IS":
                    if (drl_action < 0.999): # the drl action is not for NDE purpose
                        self.drl_epsilon_value = drl_action
                        self.real_epsilon_value = underline_drl_action
                        drl_action = self.real_epsilon_value
                    else: # Else, stay with the drl action
                        self.drl_epsilon_value = drl_action
                        self.real_epsilon_value = drl_action
                else:
                    self.drl_epsilon_value = drl_action
                    self.real_epsilon_value = drl_action

        for i in range(len(controlled_bvs_list)):
            bv = controlled_bvs_list[i]
            bv_criticality = bv_criticality_list[i]
            bv_criticality_array = criticality_array_list[i]
            bv_pdf = bv.controller.get_NDD_possi()
            combined_bv_criticality_array = bv_criticality_array
            bv_action_idx, weight, ndd_possi, critical_possi, single_weight_list = bv.controller.Decompose_sample_action(np.sum(combined_bv_criticality_array), combined_bv_criticality_array, bv_pdf, drl_action)
            if bv_action_idx is not None: 
                bv_action_idx = bv_action_idx.item()
            bv_action_idx_list.append(bv_action_idx), weight_list.append(weight), ndd_possi_list.append(ndd_possi), IS_possi_list.append(critical_possi), bv_action_criticality_list.append(bv_criticality_array[bv_action_idx])
            if single_weight_list is not None:
                whole_weight_list.append(min(single_weight_list))
            else:
                whole_weight_list.append(None)

        vehicle_criticality_list = deepcopy(bv_criticality_list)
        # Select the top bvs with highest criticality
        selected_bv_idx = sorted(range(len(bv_criticality_list)),
                                key=lambda i: bv_criticality_list[i])[-num_controlled_critical_bvs:]
        for i in range(len(controlled_bvs_list)):
            if i in selected_bv_idx: 
                if whole_weight_list[i] and whole_weight_list[i]*self.env.info_extractor.episode_log["weight_episode"]*self.env.initial_weight < conf.weight_threshold:
                    bv_action_idx_list[i], weight_list[i], ndd_possi_list[i], IS_possi_list[i], bv_action_criticality_list[i] = None, None, None, None, None
            if i not in selected_bv_idx:
                bv_action_idx_list[i], weight_list[i], ndd_possi_list[i], IS_possi_list[i], bv_action_criticality_list[i] = None, None, None, None, None
        if len(bv_criticality_list):
            max_vehicle_criticality = np.max(bv_criticality_list)
        else:
            max_vehicle_criticality = -np.inf

        return bv_action_idx_list, weight_list, max_vehicle_criticality, ndd_possi_list, IS_possi_list, controlled_bvs_list, vehicle_criticality_list, discriminator_input, bv_action_criticality_list

    @staticmethod
    # @profile
    def _get_Surrogate_CAV_action_probability(cav_obs):
        """Obtain the lane change probability of CAV. If ADS will not immediately crash, then the lane change probability is at least epsilon_lane_change_prob map gain from [0, 1] to lane change probability [epsilon_lane_change_prob, max_remaining_LC_prob].

        Args:
            cav_obs (dict): Observation of the CAV.

        Returns:
            float: Left-lane-change possibility.
            float: Stay-still possibility.
            float: Right-lane-change possibility.
        """
        CAV_left_prob, CAV_right_prob = 0, 0
        CAV_still_prob = conf.epsilon_still_prob
        left_gain, right_gain = 0, 0
        left_LC_safety_flag, right_LC_safety_flag = False, False
        # CAV will do lane change or not?
        lane_index_list = [-1, 1]  # -1: right turn; 1: left turn
        for lane_index in lane_index_list:
            LC_safety_flag, gain = NADEBVGlobalController._Mobil_surraget_model(
                cav_obs=cav_obs, lane_index=lane_index)
            # left_LC_safety_flag, gain = IDMAVController.mobil_gain(lane_index=lane_index, cav_obs = cav_obs)
            if gain is not None:
                if lane_index == -1: 
                    right_gain = np.clip(gain, 0., None)
                    right_LC_safety_flag = LC_safety_flag
                elif lane_index == 1:
                    left_gain = np.clip(gain, 0., None)
                    left_LC_safety_flag = LC_safety_flag
        assert(left_gain >= 0 and right_gain >= 0)

        # ! quick fix the CAV lane change at one side result
        if not cav_obs["Ego"]["could_drive_adjacent_lane_left"]:
            left_LC_safety_flag = 0
            left_gain = 0
        elif not cav_obs["Ego"]["could_drive_adjacent_lane_right"] == 0:
            right_LC_safety_flag = 0
            right_gain = 0

        # epsilon LC probability if no safety issue and feasible for LC
        CAV_left_prob += conf.epsilon_lane_change_prob*left_LC_safety_flag
        CAV_right_prob += conf.epsilon_lane_change_prob*right_LC_safety_flag

        max_remaining_LC_prob = 1-conf.epsilon_still_prob-CAV_left_prob-CAV_right_prob

        total_gain = left_gain+right_gain
        obtained_LC_prob_for_sharing = np.clip(utils.remap(total_gain, [0, conf.SM_MOBIL_max_gain_threshold], [
                                               0, max_remaining_LC_prob]), 0, max_remaining_LC_prob)
        CAV_still_prob += (max_remaining_LC_prob - 
                            obtained_LC_prob_for_sharing)

        if total_gain > 0:
            CAV_left_prob += obtained_LC_prob_for_sharing * \
                (left_gain/(left_gain + right_gain))
            CAV_right_prob += obtained_LC_prob_for_sharing * \
                (right_gain/(left_gain + right_gain))

        assert(0.99999 <= (CAV_left_prob + CAV_still_prob + CAV_right_prob) <= 1.0001)

        return CAV_left_prob, CAV_still_prob, CAV_right_prob

    @staticmethod
    # @profile
    def _Mobil_surraget_model(cav_obs, lane_index):
        """Apply the Mobil surrogate model to calculate the gain for this lane change maneuver. If it does not have safety issue, then return True, gain; otherwise False, None.

        Args:
            cav_obs (dict): Observation of the CAV.
            lane_index (integer): Candidate lane for the lane change behavior.

        Returns:
            bool: Safety flag.
            float: Gain.
        """
        gain = None
        cav_info = cav_obs['Ego']

        # Is the maneuver unsafe for the new following vehicle?
        if lane_index == -1:  # right turn
            new_preceding = cav_obs["RightLead"]
            new_following = cav_obs["RightFoll"]
        if lane_index == 1:  # left turn
            new_preceding = cav_obs["LeftLead"]
            new_following = cav_obs["LeftFoll"]
    
        # Check whether will crash immediately
        r_new_preceding, r_new_following = 99999, 99999
        if new_preceding:
            r_new_preceding = new_preceding["distance"]
        if new_following:
            r_new_following = new_following["distance"]
        if r_new_preceding <= 0 or r_new_following <= 0:
            return False, gain

        new_following_a = utils.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = utils.acceleration(
            ego_vehicle=new_following, front_vehicle=cav_info)

        old_preceding = cav_obs["Lead"]
        old_following = cav_obs["Foll"]
        self_pred_a = utils.acceleration(
            ego_vehicle=cav_info, front_vehicle=new_preceding)

        # The deceleration of the new following vehicle after the the LC should not be too big (negative)
        if new_following_pred_a < -conf.Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return True, 0

        # Is there an acceleration advantage for me and/or my followers to change lane?
        self_a = utils.acceleration(
            ego_vehicle=cav_info, front_vehicle=old_preceding)
        old_following_a = utils.acceleration(
            ego_vehicle=old_following, front_vehicle=cav_info)
        old_following_pred_a = utils.acceleration(
            ego_vehicle=old_following, front_vehicle=old_preceding)
        gain = self_pred_a - self_a + conf.Surrogate_POLITENESS * \
            (new_following_pred_a - new_following_a + 
            old_following_pred_a - old_following_a)
        return True, gain

    # @profile
    def map_cav_obs(self, cav_obs):
        """Map the CAV observation to the straight lanes.

        Args:
            cav_obs (dict): Observation of the CAV.

        Returns:
            dict: Mapped observation of the CAV.
        """
        new_cav_obs = deepcopy(cav_obs)
        if new_cav_obs["could_drive_adjacent_lane_left"] and not new_cav_obs["could_drive_adjacent_lane_right"]:
            new_cav_lane_index = 0
        elif new_cav_obs["could_drive_adjacent_lane_right"] and not new_cav_obs["could_drive_adjacent_lane_left"]:
            new_cav_lane_index = 1
        else:
            raise ValueError("wrong cav change lane info")
        new_cav_obs["lane_index"] = new_cav_lane_index
        original_lane_width = new_cav_obs["lane_width"]
        new_cav_obs["lateral_offset"] = new_cav_obs["lateral_offset"] * \
            4.0/original_lane_width
        new_cav_obs["lateral_speed"] = new_cav_obs["lateral_speed"] * \
            4.0/original_lane_width
        new_cav_x = 600
        new_cav_y = conf.lane_list[new_cav_lane_index] + \
            new_cav_obs["lateral_offset"]
        new_cav_obs["position"] = (new_cav_x, new_cav_y)
        return new_cav_obs

    # @profile
    def map_bv_obs(self, mapped_cav_obs, bv_obs):
        """Map the BV observation to the straight lanes centered at CAV.

        Args:
            mapped_cav_obs (dict): Mapped observation of the CAV.
            bv_obs (dict): Observation of the BV.

        Returns:
            dict: Mapped observation of the BV.
        """
        new_bv_obs = bv_obs
        bv_lane_index = mapped_cav_obs["lane_index"] + \
            bv_obs["relative_lane_index"]
        if bv_lane_index != 0 and bv_lane_index != 1:
            raise ValueError("invalid bv lane index")
        new_bv_obs["lane_index"] = bv_lane_index
        original_lane_width = new_bv_obs["lane_width"]
        new_bv_obs["lateral_offset"] = new_bv_obs["lateral_offset"] * \
            4.0/original_lane_width
        new_bv_obs["lateral_speed"] = new_bv_obs["lateral_speed"] * \
            4.0/original_lane_width
        new_bv_x = mapped_cav_obs["position"][0] + bv_obs["range"]
        new_bv_y = conf.lane_list[bv_lane_index] + new_bv_obs["lateral_offset"]
        new_bv_obs["position"] = (new_bv_x, new_bv_y)
        return new_bv_obs

    # @profile
    def get_full_obs_from_cav_obs_and_bv_list(self, CAV_obs, bv_list):
        """Get the full observation from the CAV observation and the BV list.

        Args:
            CAV_obs (dict): Observation of the CAV.
            bv_list (list): List of BVs.

        Returns:
            dict: Full observation of all the vehicles.
        """
        # This observation will be a dict containing CAV and all BV NADE candidates
        full_obs = collections.OrderedDict()
        full_obs["CAV"] = CAV_obs["Ego"]
        vehicle_id_list = [vehicle.id for vehicle in bv_list]
        cav_context = self.env.vehicle_list["CAV"].observation.context
        cav_surrounding = self._process_cav_context(vehicle_id_list)
        if conf.simulation_config["map"] == "ACM" or conf.simulation_config["map"] == "Mcity": 
            full_obs["CAV"] = self.map_cav_obs({**full_obs["CAV"], **cav_surrounding["CAV"]})
        realtime_veh_list = self.env.simulator.get_vehID_list()
        for vehicle in bv_list:
            vehicle_id = vehicle.id
            if vehicle_id in cav_context.keys() and vehicle_id in realtime_veh_list:
                full_obs[vehicle_id] = {
                    "veh_id": vehicle_id,
                    "could_drive_adjacent_lane_left": self.env.simulator.get_vehicle_lane_adjacent(vehicle_id,1),
                    "could_drive_adjacent_lane_right": self.env.simulator.get_vehicle_lane_adjacent(vehicle_id,-1),
                    "distance": 0,
                    "heading": cav_context[vehicle_id][67],
                    "lane_index": cav_context[vehicle_id][82],
                    "lateral_speed": cav_context[vehicle_id][50],
                    "lateral_offset": cav_context[vehicle_id][184],
                    "prev_action": vehicle.controller.action,
                    "position": cav_context[vehicle_id][66],
                    "position3D": cav_context[vehicle_id][57],
                    "velocity": cav_context[vehicle_id][64],
                    "road_id": cav_context[vehicle_id][80],
                    "acceleration": cav_context[vehicle_id][114]
                }
                if conf.simulation_config["map"] == "ACM" or conf.simulation_config["map"] == "Mcity": 
                    full_obs[vehicle_id] = self.map_bv_obs(full_obs["CAV"], {**full_obs[vehicle_id], **cav_surrounding[vehicle_id]})

        for veh_id in full_obs:
            if veh_id in self.vehicle_history_traj.keys():
                full_obs[veh_id]["history"] = self.vehicle_history_traj[veh_id]
        return full_obs

    # @profile
    def _process_cav_context(self, vehicle_id_list):
        """Process the CAV context.

        Args:
            vehicle_id_list (list): List of vehicle IDs.

        Returns:
            dict: Processed CAV context.
        """
        cav = self.env.vehicle_list["CAV"]
        cav_pos = cav.observation.local["CAV"][66]
        cav_context = cav.observation.context
        cav_surrounding = {}
        cav_surrounding["CAV"] = {
            "range": 0,
            "lane_width": self.env.simulator.get_vehicle_lane_width("CAV"),
            "lateral_offset": cav.observation.local["CAV"][184],
            "lateral_speed": cav.observation.local["CAV"][50],
            "position": cav_pos,
            "prev_action": cav.observation.information["Ego"]["prev_action"],
            "relative_lane_index": 0,
            "speed": cav.observation.local["CAV"][64]
        }
        total_vehicle_id_list = list(
            set(vehicle_id_list) | set(cav_context.keys()))
        # print(total_vehicle_id_list)
        real_veh_list = self.env.simulator.get_vehID_list()
        for veh_id in total_vehicle_id_list:
            if veh_id in real_veh_list:
                bv_pos = cav_context[veh_id][66]
                # distance = self.env.simulator.get_vehicles_dist(cav_pos,bv_pos)
                distance = self.env.simulator.get_vehicles_dist_road("CAV", veh_id)

                if distance > conf.cav_obs_range+5:
                    # distance_alter = self.env.simulator.get_vehicles_dist(bv_pos,cav_pos)
                    distance_alter = self.env.simulator.get_vehicles_dist_road(
                        veh_id, "CAV")
                    if distance_alter > conf.cav_obs_range+5:
                        continue
                        # distance = 2
                        # relative_lane_index = -self.env.simulator.get_vehicles_relative_lane_index(veh_id,"CAV")
                    else:
                        distance = -distance_alter
                        relative_lane_index = - \
                            self.env.simulator.get_vehicles_relative_lane_index(
                                veh_id, "CAV")
                else:
                    relative_lane_index = self.env.simulator.get_vehicles_relative_lane_index(
                        "CAV", veh_id)
                try:
                    prev_action = self.env.vehicle_list[veh_id].observation.information["Ego"]["prev_action"]
                except:
                    prev_action = None
                cav_surrounding[veh_id] = {
                    "range": distance,
                    "lane_width": self.env.simulator.get_vehicle_lane_width(veh_id),
                    "lateral_offset": cav_context[veh_id][184],
                    "lateral_speed": cav_context[veh_id][50],
                    "position": bv_pos,
                    "prev_action": prev_action,
                    "relative_lane_index": relative_lane_index,
                    "speed": cav_context[veh_id][64]
                }
        return cav_surrounding
