import bisect
import json, ujson
import math
import numpy as np
import os
import pandas as pd
from operator import itemgetter
from itertools import groupby
import time
from mtlsp.envs.env import BaseEnv
from mtlsp.controller.vehicle_controller.controller import Controller
from mtlsp.controller.vehicle_controller.idmcontroller import IDMController
from mtlsp.controller.vehicle_controller.globalcontroller import DummyGlobalController
from mtlsp.vehicle.vehicle import Vehicle, VehicleDynamics
from mtlsp.logger.infoextractor import InfoExtractor
from mtlsp import utils as mtlsp_utils
if os.environ["mode"] == "testing":
    import conf.conf_testing as conf
elif os.environ["mode"] == "training":
    import conf.conf_training as conf
else:
    raise ValueError("Please set the mode to testing or training")
from controller.nddcontroller import NDDController
from controller.nddglobalcontroller import NDDBVGlobalController
import utils


counterfactual_time = 2


class NDE(BaseEnv):
    def __init__(self, AVController=IDMController, BVController=NDDController, AVGlobalController=DummyGlobalController, BVGlobalController=NDDBVGlobalController, info_extractor=InfoExtractor, local_flag=True):
        """Initialize the NDE class.

        Args:
            AVController (object): AV controller.
            BVController (object): BV controller.
            AVGlobalController (object): AV global controller.
            BVGlobalController (object): BV global controller.
            info_extractor (object): Information extractor.
            local_flag (bool): Flag of whether only the CAV's surrounding vehicles are using the NDD controller.
        """
        self.default_av_controller = AVController
        self.ndd_local_flag = local_flag
        super().__init__(
            global_controller_dict={
                "BV": BVGlobalController, "CAV": AVGlobalController},
            independent_controller_dict={
                "BV": BVController, "CAV": AVController},
            info_extractor=info_extractor
        )
        self.cav_action = 20
        self.cav_initial_states = None
        self.cav_initial_lateral_speed = None
        self.init_time_step = None
        self.cav_initial_RSS_states = None
        self.cav_initial_obs = None
        self.cav_initial_neuralmetric_obs = None
        self.replay_trajs = None
        self.replay_json_obj = None
        self.last_nade_info = None
        self.last_second = False
        self.eval_flag = False
        self.soft_reboot_time_section = None
        self.three_circle_min_distance = np.inf

    def initialize(self):
        """Initialize the NDE simulation.
        """
        super().initialize()
        self.cav_action = 20
        self.cav_initial_states = None
        self.cav_initial_lateral_speed = None
        self.init_time_step = None
        self.cav_initial_RSS_states = None
        self.cav_initial_obs = None
        self.replay_trajs = None
        self.replay_json_obj = None
        self.last_nade_info = None
        self.soft_reboot()
        for veh_id in self.simulator.get_vehID_list():
            self.simulator.set_vehicle_max_lateralspeed(veh_id, 4.0)

    # @profile
    def _step(self, action=None):
        """NDE simulation step.

        Args:
            action (np.array): Action of the CAV. Defaults to None.        
        """
        self.cav_action = action
        if action is None:
            pass
            # print("Find!")
        # print(self.simulator.get_time())
        return super()._step()

    def add_background_vehicles(self, vlist, add_to_vlist=True, add_to_sumo=True):
        """Add background vehicles to the simulation.

        Args:
            vlist (list): List of vehicles.
            add_to_vlist (bool, optional): Flag of whether to add the vehicles to the vehicle list. Defaults to True.
            add_to_sumo (bool, optional): Flag of whether to add the vehicles to the SUMO simulation. Defaults to True.
        """
        if add_to_vlist:
            self.vehicle_list.add_vehicles(vlist)
        if add_to_sumo:
            for v in vlist:
                # self.simulator._add_vehicle_to_sumo(v)
                # ! should be customized later
                self.simulator._add_vehicle_to_sumo(v, typeID='IDM')

    def generate_av(self, speed=30.0, id="CAV", route="route_0", type_id="IDM_CAV", position=400.0, av_lane_id=None, controller_type=IDMController):
        """Generate one av in the network.

        Args:
            speed (float, optional): Initial speed. Defaults to 0.0.
            id (string, optional): CAV ID. Defaults to "CAV".
            route (string, optional): Route ID. Defaults to "route_0".
            type_id (string, optional): Vehicle type ID. Defaults to "IDM".
            position (float, optional): Initial position of the vehicle. Defaults to 400.0.
            controller_type (class, optional): Controller type of the AV. Defaults to AVController.

        Returns:
            str: Vehicle ID.
            dict: Vehicle information.
        """
        if av_lane_id is None:
            if conf.experiment_config["train_flag"]:
                index = np.random.randint(3)
            else:
                index = 1
            av_lane_id = self.simulator.get_available_lanes()[index].getID()
            # print(av_lane_id)
        av = VehicleDynamics(id=id, controller=self.default_av_controller(
        ), routeID=route, simulator=self.simulator, initial_speed=speed, initial_position=position, initial_lane_id=av_lane_id)
        self.simulator._add_vehicle_to_sumo(av, typeID=type_id)
        av.install_controller(controller_type())
        self.vehicle_list.add_vehicles([av])
        info = {
            'speed': speed,
            'lane_id': av_lane_id,
            'route_id': route,
            'position': position
        }
        return id, info

    def generate_bv_traffic_flow(self):
        """Generate background vehicles in the network.
        """
        for lane in self.simulator.get_available_lanes():
            self.generate_ndd_flow_on_single_lane(lane_id=lane.getID())

    def generate_traffic_flow(self, init_info=None):
        """Generate traffic flow including one AV abd several BVs based on NDD.
        """
        if init_info is None:
            if conf.traffic_flow_config["CAV"]:
                avID, avINFO = self.generate_av(
                    controller_type=self.default_av_controller)
                if 'initialization' in self.simulator.output:
                    self.info_extractor.add_initialization_info(avID, avINFO)
            if conf.traffic_flow_config["BV"]:
                self.generate_bv_traffic_flow()
        else:
            for vehID, info in init_info.items():
                speed = info['speed']
                route = info['route_id']
                position = info['position']
                lane_id = info['lane_id']
                if 'AV' in vehID:
                    avID, avINFO = self.generate_av(
                        speed=speed,
                        id=vehID,
                        route=route,
                        position=position,
                        av_lane_id=lane_id,
                        controller_type=self.default_av_controller
                    )
                    if 'initialization' in self.simulator.output:
                        self.info_extractor.add_initialization_info(
                            avID, avINFO)
                if 'BV' in vehID:
                    self.add_background_vehicles(Vehicle(vehID, controller=DummyController(), routeID=route, simulator=self.simulator,
                                                         initial_speed=speed, initial_position=position, initial_lane_id=lane_id), add_to_vlist=False, add_to_sumo=True)
                    if 'initialization' in self.simulator.output:
                        info = {
                            'speed': speed,
                            'lane_id': lane_id,
                            'route_id': route,
                            'position': position
                        }
                        self.info_extractor.add_initialization_info(
                            vehID, info)

    def soft_reboot(self):
        """Delete all vehicles and re-generate all vehicles on the road.
        """
        # ! tmp disable the delete vehicle capacity
        # self.delete_all_vehicles()
        if self.simulator.input is None:
            self.generate_traffic_flow()
        elif 'initialization' in self.simulator.input:
            if os.path.isdir(self.simulator.input_path):
                init_file = str(self.simulator.episode)+'.init.json'
                init_file_path = os.path.join(
                    self.simulator.input_path, init_file)
                if os.path.isfile(init_file_path):
                    print('Initialize from existed initialization file!')
                    with open(init_file_path, 'r') as json_file_init:
                        init_info = ujson.load(json_file_init)
                    self.generate_traffic_flow(init_info)
                else:
                    print('Re-initialize randomly!')
                    self.generate_traffic_flow()
            else:
                raise ValueError('The input does not exist!')
        elif self.simulator.replay_flag:
            vehicle_info = self.simulator.input_trajs[0.0]
            for vehID in vehicle_info.keys():
                # add vehicle
                individual_info = vehicle_info[vehID]
                if float(individual_info['speed']) > conf.v_max:
                    print("speed too large:",individual_info)
                    individual_info['speed'] = str(conf.v_max)
                v = Vehicle(vehID, controller=Controller(), routeID='route_0',
                            simulator=self.simulator, initial_speed=float(individual_info['speed']))
                type_id = individual_info['type'].split("@")[0]
                self.simulator._add_vehicle_to_sumo(v, type_id)
                self.simulator.change_vehicle_position(vehID, (float(individual_info['x']), float(
                    individual_info['y'])), angle=float(individual_info['angle']), keepRoute=0)
                self.simulator.set_vehicle_speed(
                    vehID, float(individual_info['speed']))
        elif conf.train_mode == "offline":
            # initialization for offline training
            # print(self.simulator.input[0])
            t0 = time.time()

            self.replay_trajs, self.replay_json_obj = None, None
            self.replay_trajs = mtlsp_utils.load_trajs_from_jsonfile(self.simulator.input[0:2])
            t01 = time.time()

            jsonfile_path = self.simulator.input[1]+"/"+str(self.simulator.input[0][1])+".json"
            with open(jsonfile_path) as json_file:
                line = mtlsp_utils.get_line(json_file, self.simulator.input[0][3])
            self.replay_json_obj = ujson.loads(line)
            # assert(self.replay_trajs is not None and self.replay_json_obj is not None)
            t1 = time.time()

            cav_info = self.replay_json_obj["CAV_info"]
            self.init_time_step = list(self.replay_trajs.keys())[0]
            if conf.precise_criticality_flag:
                if conf.precise_criticality_threshold != 0:
                    ranges = [self.simulator.input[0][4], self.simulator.input[0][5]]
                    if self.simulator.input[3] == 0: # safe replay
                        self.init_time_step = str(round(ranges[0]/10,1))
                        self.end_time_step = str(round(ranges[1]/10,1))
                    else: # crash replay
                        self.init_time_step = str(round(ranges[0]/10,1))
                        # self.end_time_step = str(round(ranges[-1][-1]/10,1))
                    if self.init_time_step not in cav_info.keys():
                        self.init_time_step = list(cav_info.keys())[-1]
                elif conf.precise_weight_threshold != 0:
                    weight_list = np.array(list(self.replay_json_obj["weight_list_step"].values()))
                    cum_weight_list = np.cumprod(weight_list)
                    find_flag = False
                    for i in range(len(cum_weight_list)):
                        if cum_weight_list[i] < conf.precise_weight_threshold:
                            find_flag = True
                            break
                    if find_flag:
                        self.init_time_step = list(self.replay_json_obj["weight_list_step"].keys())[i]
                    else:
                        self.init_time_step = list(self.replay_trajs.keys())[-1]
                    self.end_time_step = list(self.replay_trajs.keys())[-1]

            if self.simulator.input[3] != 0:
                self.end_time_step = list(self.replay_trajs.keys())[-1]
                current_nade_lc_time_index = None
                for time_index in self.replay_json_obj["NADE_info"].keys():
                    nade_action = list(self.replay_json_obj["NADE_info"][time_index].values())[0][0]
                    if nade_action in [0,1]:
                        if not current_nade_lc_time_index:
                            current_nade_lc_time_index = time_index
                            if float(self.end_time_step)-float(time_index) < 1:
                                last_nade_action_time = time_index
                                self.last_nade_info = self.replay_json_obj["NADE_info"][last_nade_action_time]
                                self.end_time_step = str(round(float(last_nade_action_time)+counterfactual_time,1))
                                break
                        elif float(time_index)-float(current_nade_lc_time_index) >= 0.9999:
                            current_nade_lc_time_index = time_index
                            if float(self.end_time_step)-float(time_index) < 1:
                                last_nade_action_time = time_index
                                self.last_nade_info = self.replay_json_obj["NADE_info"][last_nade_action_time]
                                self.end_time_step = str(round(float(last_nade_action_time)+counterfactual_time,1))
                                break
                if round(float(self.end_time_step),1)-round(float(self.init_time_step),1) < counterfactual_time:
                    self.end_time_step = str(round(float(self.init_time_step)+counterfactual_time,1))
            # print(self.init_time_step, self.end_time_step)
            init_info = self.replay_trajs[self.init_time_step]
            av_information = init_info["CAV"]
            if self.init_time_step not in cav_info:
                print(self.simulator.input)
                print(self.init_time_step, cav_info.keys())
                print(self.replay_json_obj["weight_list_step"])
            self.cav_initial_states = cav_info[self.init_time_step]["CAV_action"]["additional_info"]["veh_states"]
            # print(cav_info[self.init_time_step]["CAV_action"]["additional_info"][0])
            self.cav_initial_lateral_speed = mtlsp_utils.remap(cav_info[self.init_time_step]["CAV_action"]["additional_info"]["rl_obs"][1], [-1, 1], [-conf.lat_v_max, conf.lat_v_max])
            self.cav_initial_RSS_states = {}
            for veh_id in cav_info[self.init_time_step]["CAV_action"]["additional_info"]["RSS_states"]:
                self.cav_initial_RSS_states[veh_id] = dict(cav_info[self.init_time_step]["CAV_action"]["additional_info"]["RSS_states"][veh_id])
            self.cav_initial_obs = list(cav_info[self.init_time_step]["CAV_action"]["additional_info"]["rl_obs"])
            if conf.simulation_config["neuralmetric_flag"]:
                self.cav_initial_neuralmetric_obs = list(cav_info[self.init_time_step]["CAV_action"]["additional_info"]["NN_metric_obs"])
            
            av = VehicleDynamics(
                "CAV", controller=self.default_av_controller(), routeID='route_0', simulator=self.simulator, initial_speed=float(av_information['speed'])
            )
            type_id = av_information['type'].split("@")[0]
            self.simulator._add_vehicle_to_sumo(av, type_id)
            self.simulator.change_vehicle_position(
                "CAV", (float(av_information['x']), float(av_information['y'])), angle=float(av_information['angle']), keepRoute=0
            )
            self.simulator.set_vehicle_speed("CAV", float(av_information['speed']))
            # av.install_controller(self.default_av_controller())
            # self.vehicle_list.add_vehicles([av])
            for vehID in init_info.keys():
                individual_info = init_info[vehID]
                if vehID != "CAV":
                    v = Vehicle(
                        vehID, controller=Controller(), routeID='route_0', simulator=self.simulator, initial_speed=min(float(individual_info['speed']),40.0)
                    )
                    type_id = individual_info['type'].split("@")[0]
                    self.simulator._add_vehicle_to_sumo(v, type_id)
                    self.simulator.change_vehicle_position(
                        vehID, (float(individual_info['x']), float(individual_info['y'])), angle=float(individual_info['angle']), keepRoute=0
                    )
                    self.simulator.set_vehicle_speed(vehID, float(individual_info['speed']))
                
            t2 = time.time()
            self.soft_reboot_time_section = [t1-t0,t2-t1,t01-t0,t1-t01]

        # Focus on the CAV in the SUMO-GUI
        if self.simulator.track_cav:
            self.simulator.track_vehicle_gui()
            self.simulator.set_zoom(500)
            self.simulator.set_gui_schema("real world")

    @staticmethod
    def generate_random_vehicle():
        """Generate a random vehicle at the beginning of the road in the free-flow mode.

        Returns:
            float: Speed of the vehicle.
            float: Position of the vehicle.
        """ 
        random_number = np.random.uniform()
        idx = bisect.bisect_left(conf.speed_CDF, random_number)
        # exposure_freq_speed = conf.speed_CDF[idx] - conf.speed_CDF[idx-1] if idx >= 1 else conf.speed_CDF[idx]
        speed = conf.v_to_idx_dic.inverse[idx]
        rand_position = round(np.random.uniform(
            conf.random_veh_pos_buffer_start, conf.random_veh_pos_buffer_end))
        # exposure_freq = exposure_freq_speed * 1/(conf.random_veh_pos_buffer_start - conf.random_veh_pos_buffer_end)
        return speed, rand_position

    @staticmethod
    def sample_CF_FF_mode():
        """Randomly choose the car-following or free-flow mode to generate vehicles.

        Returns:
            str: Mode ID.
        """
        random_number_CF = np.random.uniform()
        if random_number_CF > conf.CF_percent:
            return "FF"
        else:
            return "CF"

    def generate_FF_vehicle(self, back_speed_position=None, direction=1):
        """Generate vehicle in free-flow mode, the vehicle driving behind is needed.

        Args:
            back_vehicle (dict, optional): Speed and position of the vehicle driving behind. Defaults to None.

        Returns:
            dict: Return the speed and position of the newly created vehicle.
        """
        if back_speed_position is not None:
            rand_speed, rand_position = self.generate_random_vehicle()
            pos_generate = back_speed_position["position"] + \
                (conf.ff_dis + rand_position + conf.LENGTH)*direction
            return {"speed": rand_speed, "position": pos_generate}
        else:
            raise Exception(
                "Warning: generating FF vehicle with no back vehicle")

    def generate_CF_vehicle(self, back_speed_position=None, direction=1):
        """Generate vehicles in the car-following scenario.

        Args:
            back_speed_position (dict, optional): Speed and position of the last created vehicle. Defaults to None.

        Returns:
            dict: New created vehicle information including speed, and position.
        """
        if back_speed_position["speed"] < conf.v_low:
            presum_list = conf.presum_list_forward[conf.v_to_idx_dic[conf.v_low]]
        else:
            presum_list = conf.presum_list_forward[conf.v_to_idx_dic[int(
                back_speed_position["speed"])]]
        random_number = np.random.uniform()
        r_idx, rr_idx = divmod(bisect.bisect_left(
            presum_list, random_number), conf.num_rr)
        try:
            r, rr = conf.r_to_idx_dic.inverse[r_idx], conf.rr_to_idx_dic.inverse[rr_idx]
        except:
            if back_speed_position["speed"] > 35:
                r, rr = 50, -2
            else:
                r, rr = 50, 2
        # Accelerated training for initialization
        r = r - conf.Initial_range_adjustment_AT + conf.Initial_range_adjustment_SG
        if r <= conf.Initial_range_adjustment_SG:
            r = r + conf.r_high
        speed = back_speed_position["speed"] + rr
        position = back_speed_position["position"] + \
            (r + conf.LENGTH)*direction
        return {"speed": speed, "position": position}

    def generate_ndd_vehicle(self, back_speed_position=None, lane_id=None):
        """This function will generate an vehicle under NDD distribution.

        Args:
            back_speed_position (dict, optional): Information of the last created background vehicle. Defaults to None.
        """
        if back_speed_position == None:
            speed, position = self.generate_random_vehicle()
        else:
            mode = self.sample_CF_FF_mode()
            if mode == "FF":
                speed_position = self.generate_FF_vehicle(back_speed_position)
                speed, position = speed_position["speed"], speed_position["position"]
            elif mode == "CF":
                speed_position = self.generate_CF_vehicle(back_speed_position)
                speed, position = speed_position["speed"], speed_position["position"]
            else:
                raise ValueError(
                    f"The vehicle mode needs to be CF/FF, however {mode} detected")
        vehID = mtlsp_utils.generate_unique_bv_id()
        route = 'route_0'  # !implement hard code here
        if speed > conf.v_high:
            speed = conf.v_high
        if speed < conf.v_low:
            speed = conf.v_low
        self.add_background_vehicles(Vehicle(vehID, controller=Controller(), routeID=route, simulator=self.simulator,
                                             initial_speed=speed, initial_position=position, initial_lane_id=lane_id), add_to_vlist=False, add_to_sumo=True)
        if 'initialization' in self.simulator.output:
            info = {
                'speed': speed,
                'lane_id': lane_id,
                'route_id': route,
                'position': position
            }
            self.info_extractor.add_initialization_info(vehID, info)
        return {"speed": speed, "position": position}

    def generate_ndd_flow_on_single_lane(self, lane_id):
        """Generate NDD vehicle flow one one single lane.

        Args:
            lane_id (str): Lane ID.
        """
        if ("CAV" not in self.vehicle_list) or self.vehicle_list["CAV"].initial_lane_id != lane_id:
            # This lane does not have CAV as an initialization
            # ! Temporarily only have "CAV" as the only vehicle, maybe changed later
            tmp_speed_position = self.generate_ndd_vehicle(
                back_speed_position=None, lane_id=lane_id)
        else:
            tmp_speed_position = {
                "speed": self.vehicle_list["CAV"].initial_speed, "position": self.vehicle_list["CAV"].initial_position}

        tmp_speed, tmp_position = tmp_speed_position["speed"], tmp_speed_position["position"]

        while tmp_position <= conf.gen_length:
            tmp_speed_position = self.generate_ndd_vehicle(
                back_speed_position=tmp_speed_position, lane_id=lane_id)
            tmp_speed, tmp_position = tmp_speed_position["speed"], tmp_speed_position["position"]

    def _terminate_check(self):
        """Check the termination condition of the simulation.

        Returns:
            dict: Reason of the termination.
            bool: Flag of whether the simulation is stopped.
            dict: Additional information.
        """
        collision = tuple(set(self.simulator.detected_crash()))
        new_crash_flag, crash_id = self.detect_crash_new()
        reason = None
        stop = False
        additional_info = {}
        cav_head_pos = self.simulator.get_vehicle_position("CAV")
        cav_heading = self.simulator.get_vehicle_heading("CAV")
        extra_buff = 0.1
        cav_length, cav_width = 5.0, 1.8
        cav_center_pos = [cav_head_pos[0]-cav_length/2*np.sin(
            cav_heading/180*np.pi), cav_head_pos[1]-cav_length/2*np.cos(cav_heading/180*np.pi)]
        sumoT = self.simulator.get_time()
        if new_crash_flag:
            reason = {1: "CAV and BV collision"}
            stop = True
            additional_info = {"collision_id": crash_id}
        elif "CAV" not in self.vehicle_list:
            reason = {2: "CAV leaves network"}
            stop = True
        elif self.simulator.detect_vehicle_num() == 0:
            reason = {3: "All vehicles leave network"}
            stop = True
        elif self.simulator.get_vehicle_position("CAV")[0] > 800.0:
            reason = {4: "CAV reaches 800 m"}
            stop = True
        elif bool(collision) and "CAV" not in collision:
            reason = {5: "BV and BV Collision"}
            stop = True
        elif cav_center_pos[1] > 52.0-cav_width/2+extra_buff or cav_center_pos[1] < 40.0+cav_width/2-extra_buff:
            reason = {6: "CAV leaves lane"}
            stop = True
        # elif self.simulator.get_vehicle_speed("CAV") < 20.0:
        #     reason = {7: "CAV too slow"}
        #     stop = True
        # elif self.simulator.get_vehicle_speed("CAV") > 40.0:
        #     reason = {8: "CAV too fast"}
        #     stop = True
        elif conf.train_mode == "offline":
            if round(sumoT+float(self.init_time_step),1) > round(float(self.end_time_step),1):
                stop = True
                reason = {9: "Replay ends"}
            # elif counterfactual_time > 1 and round(sumoT+float(self.init_time_step),1) > round(float(self.end_time_step)-(counterfactual_time-1),1):
            #     self.last_second = True
        if stop:
            self.episode_info["end_time"] = sumoT-self.simulator.step_size
            # print(reason, self.vehicle_list["CAV"].observation.local)
        return reason, stop, additional_info

    def get_observation_drl(self):
        """Get the observation for the CAV agent.

        Returns:
            np.array: Observation of CAV.
        """
        cav = self.vehicle_list["CAV"]
        cav_info = cav.observation.local["CAV"]
        context_info = cav.observation.context
        if self.simulator.sumo_time_stamp == 0 and conf.train_mode == "offline":
            return np.array(self.cav_initial_obs)
        else:
            return NDE._transfer_obs_to_state(cav_info, context_info, self.simulator, cav)

    def get_observation_neuralmetric(self):
        """Get the observation for the neural metric.

        Returns:
            np.array: Observation of the neural metric.
        """
        cav = self.vehicle_list["CAV"]
        cav_info = cav.observation.local["CAV"]
        context_info = cav.observation.context
        if self.simulator.sumo_time_stamp == 0 and conf.train_mode == "offline":
            return np.array(self.cav_initial_neuralmetric_obs).reshape(1,-1)
        else:
            return NDE._transfer_obs_to_neuralmetricstate(cav_info, context_info, self.simulator, cav)

    @staticmethod
    # @profile
    def _transfer_obs_to_state(cav_info, cav_context, simulator, cav):
        """Transfer the observation to the input for the CAV agent.

        Args:
            cav_info (dict): CAV information.
            cav_context (dict): Context information.
            simulator (object): SUMO simulator.
            cav (object): CAV vehicle.

        Returns:
            np.array: Input of the CAV agent.
        """
        veh_length = 5
        cav_pos = cav_info[66]
        ego_state = [
            # Normalized cav longitudinal speed
            min(mtlsp_utils.remap(cav_info[64], [0, conf.v_max], [-1, 1]), 1),
            # Normalized cav lateral speed
            mtlsp_utils.remap(cav.lateral_speed,
                        [-conf.lat_v_max, conf.lat_v_max], [-1, 1]),
            # mtlsp_utils.remap(cav_info[82], [0,2], [-1,1]), # Normalized lane index
            # Normalized distance to the leftmost lane marker
            mtlsp_utils.remap(52-cav_info[66][1], [-2, 14], [-1, 1])
        ]
        ego_state = np.array(ego_state)

        list_full = []
        for veh_id in cav_context.keys():
            bv_pos = cav_context[veh_id][66]
            distance_long = bv_pos[0]-cav_pos[0]
            distance_lat = bv_pos[1]-cav_pos[1]
            if abs(distance_long) <= conf.cav_obs_range:
                new_row = [
                    # veh_id,  # ID
                    mtlsp_utils.remap(distance_long, [-conf.cav_obs_range, conf.cav_obs_range], [-1, 1]),  # range
                    min(mtlsp_utils.remap(cav_context[veh_id][64], [0, conf.v_max], [-1, 1]), 1),  # Speed_longitudinal
                    mtlsp_utils.remap(cav_context[veh_id][50], [-conf.lat_v_max, conf.lat_v_max], [-1, 1]),  # Speed_lateral
                    mtlsp_utils.remap(distance_lat, [-12, 12], [-1, 1]),  # Distance_lateral
                    mtlsp_utils.cal_euclidean_dist(cav_pos, bv_pos)  # Distance
                ]
                list_full.append(new_row)
        sorted_list_full = sorted(list_full, key=lambda x: (x[4]))
        # print(sorted_list_full)
        sorted_array_full = np.array(sorted_list_full)
        if len(sorted_list_full) < conf.cav_observation_num:
            # at position 0, with minimum speed and at the top lane
            fake_vehicle_row = [-1.0, -1.0, 0.0, 1.0]
            fake_vehicle_rows = [fake_vehicle_row for _ in range(conf.cav_observation_num - len(sorted_list_full))]
            if len(sorted_list_full) == 0:
                final_array = np.array(fake_vehicle_rows)
            else:
                rows = np.array(fake_vehicle_rows)
                final_array = np.append(sorted_array_full[:,:4], rows, axis=0)
        elif len(sorted_list_full) >= conf.cav_observation_num:
            final_array = sorted_array_full[:conf.cav_observation_num, :4]
        state_origin = final_array.flatten()
        state = np.append(ego_state, np.array(
            [float(s) for s in state_origin]))
        state = state.astype('float32')
        # print(cav_context)
        # print(state)
        return state

    @staticmethod
    # @profile
    def _transfer_obs_to_neuralmetricstate(cav_info, cav_context, simulator, cav):
        """Transfer the observation to the input for the neural metric.

        Args:
            cav_info (dict): CAV information.
            cav_context (dict): Context information.
            simulator (object): SUMO simulator.
            cav (object): CAV vehicle.

        Returns:
            np.array: Input of the neural metric.
        """
        cav_x, cav_y = cav_info[66][0], cav_info[66][1]
        cav_v = cav_info[64]
        cav_heading = cav_info[67]
        ego_state = [cav_x, cav_y, cav_v, cav_heading]

        list_full = []
        for veh_id in cav_context.keys():
            bv_x, bv_y = cav_context[veh_id][66][0], cav_context[veh_id][66][1]
            bv_v = cav_context[veh_id][64]
            bv_heading = cav_context[veh_id][67]
            new_row = [
                bv_x, bv_y, bv_v, bv_heading,
                mtlsp_utils.cal_euclidean_dist((cav_x,cav_y), (bv_x,bv_y))  # Distance
            ]
            list_full.append(new_row)
        sorted_list_full = sorted(list_full, key=lambda x: (x[4]))
        # print(sorted_list_full)
        sorted_array_full = np.array(sorted_list_full)
        if len(sorted_list_full) < 6:
            # at position 0, with minimum speed and at the top lane
            fake_vehicle_row = [-1, -1, -1, -1]
            fake_vehicle_rows = [fake_vehicle_row for _ in range(6 - len(sorted_list_full))]
            if len(sorted_list_full) == 0:
                final_array = np.array(fake_vehicle_rows)
            else:
                rows = np.array(fake_vehicle_rows)
                final_array = np.append(sorted_array_full[:,:4], rows, axis=0)
        elif len(sorted_list_full) >= 6:
            final_array = sorted_array_full[:6, :4]
        state_origin = final_array.flatten()
        state = np.append(ego_state, np.array(
            [float(s) for s in state_origin]))
        state = state.reshape(1,-1)
        return state

    @staticmethod
    # @profile
    def _transfer_state_to_obs(cav_state):
        """Transfer the state to the observation for the CAV agent.

        Args:
            cav_state (np.array): State of the CAV agent.

        Returns:
            dict: Observation of the CAV agent.
        """
        normalized_cav_longitudinal_speed = cav_state[0]
        normalized_cav_lateral_speed = cav_state[1]
        normalized_distance_to_the_leftmost_lane_marker = cav_state[2]
        cav_info = {
            # lateral speed
            50: mtlsp_utils.remap(normalized_cav_lateral_speed, [-1, 1], [-conf.lat_v_max, conf.lat_v_max]),
            # longitudinal speed
            64: mtlsp_utils.remap(normalized_cav_longitudinal_speed, [-1, 1], [0, conf.v_max]),
            # position
            66: (400, np.clip(52-mtlsp_utils.remap(normalized_distance_to_the_leftmost_lane_marker, [-1, 1], [-2, 14]), 40, 52)),
        }
        cav_context = {"CAV": cav_info}
        for i in range(int((len(cav_state)-3)/4)):
            bv_state = cav_state[3+i*4:3+(i+1)*4]
            if list(bv_state) != [-1.0, -1.0, 0.0, 1.0]:
                normalized_bv_longitudinal_distance = bv_state[0]
                normalized_bv_longitudinal_speed = bv_state[1]
                normalized_bv_lateral_speed = bv_state[2]
                normalized_bv_lateral_distance = bv_state[3]
                bv_longitudinal_distance = mtlsp_utils.remap(
                    normalized_bv_longitudinal_distance, [-1, 1], [-conf.cav_obs_range, conf.cav_obs_range])
                bv_lateral_distance = mtlsp_utils.remap(
                    normalized_bv_lateral_distance, [-1, 1], [-12, 12])
                bv_info = {
                    # lateral speed
                    50: mtlsp_utils.remap(normalized_bv_lateral_speed, [-1, 1], [-conf.lat_v_max, conf.lat_v_max]),
                    # longitudinal speed
                    64: mtlsp_utils.remap(normalized_bv_longitudinal_speed, [-1, 1], [0, conf.v_max]),
                    # position
                    66: (cav_info[66][0]+bv_longitudinal_distance, np.clip(cav_info[66][1]+bv_lateral_distance, 40, 52)),
                }
                cav_context["BV_"+str(i)] = bv_info
        return cav_context

    def _replay_vehicle_list(self):
        """Replay the vehicle list in the simulation from the recorded trajectories.
        """
        # Replay module
        realtime_vehID_set = set(self.simulator.get_vehID_list())
        time_step = str(round(self.simulator.get_time()+float(self.simulator.env.init_time_step),1))
        if time_step in self.replay_trajs.keys():
            vehicle_info = self.replay_trajs[time_step]
            expected_vehID_set = set(vehicle_info.keys())
            if realtime_vehID_set != expected_vehID_set:
                for vehID in expected_vehID_set:
                    if vehID not in realtime_vehID_set:
                        # add vehicle
                        individual_info = vehicle_info[vehID]
                        v = Vehicle(vehID, controller=Controller(), routeID='route_0', simulator=self.simulator, initial_speed=float(individual_info['speed']))
                        type_id = individual_info['type'].split("@")[0]
                        self.simulator._add_vehicle_to_sumo(v, type_id)
                        self.simulator.change_vehicle_position(vehID, (float(individual_info['x']), float(individual_info['y'])), angle=float(individual_info['angle']), keepRoute=0)
                        self.simulator.set_vehicle_speed(vehID, float(individual_info['speed']))
                for vehID in realtime_vehID_set:
                    if vehID not in expected_vehID_set:
                        # delete vehicle
                        self.simulator.delete_vehicle(vehID)
    
    # @profile
    def detect_crash_new(self):
        """Detect the crash in the simulation.

        Returns:
            bool: Flag representing whether the crash is detected.
            tuple: ID tuple for the collided vehicles if any.
        """
        crash_flag = False
        crash_id_tuple = ()
        cav_info = self.vehicle_list['CAV'].observation.information
        cav_head_pos = cav_info["Ego"]["position"]
        cav_heading = cav_info["Ego"]["heading"]/180*math.pi
        cav_center_list = utils.three_circle_center_helper(cav_head_pos, cav_heading)
        num_veh = 0
        moment_min_distance_list = []
        for key in cav_info:
            if key != "Ego" and cav_info[key] != None:
                num_veh += 1
                veh_head_pos = cav_info[key]["position"]
                veh_heading = cav_info[key]["heading"]/180*math.pi
                veh_center_list = utils.three_circle_center_helper(veh_head_pos, veh_heading)
                three_circle_crash_flag, circle3_min_distance = utils.detect_crash_three_circle(cav_center_list, veh_center_list)
                moment_min_distance_list.append(circle3_min_distance)
                if three_circle_crash_flag:
                    if utils.sat_detection(cav_center_list[1], cav_heading, veh_center_list[1], veh_heading):
                        # print(cav_center_list, veh_center_list)
                        crash_flag = True
                        if crash_id_tuple == ():
                            crash_id_tuple = ("CAV",  cav_info[key]["veh_id"])
        if len(moment_min_distance_list) > 0:
            moment_min_distance = min(moment_min_distance_list)
            if moment_min_distance < self.three_circle_min_distance:
                self.three_circle_min_distance = moment_min_distance
        return crash_flag, crash_id_tuple