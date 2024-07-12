from copy import copy
import numpy as np
import math
from mtlsp.controller.vehicle_controller.controller import Controller
from mtlsp.observation.observation import Observation
import os
if os.environ["mode"] == "testing":
    import conf.conf_testing as conf
elif os.environ["mode"] == "training":
    import conf.conf_training as conf
else:
    raise ValueError("Please set the mode to testing or training")
from mtlsp.utils import check_network_boundary, update_vehicle_real_states

lane_angle = np.pi/2
max_sublane_distance_per_step = 0.4

class Vehicle(object):
    color_red = (255,0,0)
    color_yellow = (255,255,0)
    color_blue = (0,0,255)
    color_green = (0,255,0)
    color_orange = (255,165,0)
    color_purple = (255,0,255)
    color_pink = (255,165,255)

    r_low, r_high, rr_low, rr_high, acc_low, acc_high = 0, 115, -10, 8, -4, 2

    def __init__(self, id, controller, observation_method=Observation, routeID=None, simulator=None, initial_speed=None, initial_position=None, initial_lane_id=None):
        """Initialize the vehicle object.

        Args:
            id (str): Vehicle ID.
            controller (Controller): Controller object for the vehicle.
            observation_method (Observation, optional): Observation method for the vehicle. Defaults to Observation.
            routeID (str, optional): Route ID of the vehicle. Defaults to None.
            simulator (Simulator, optional): Simulator object for the vehicle. Defaults to None.
            initial_speed (float, optional): Initial speed of the vehicle. Defaults to None.
            initial_position (list, optional): Initial position of the vehicle. Defaults to None.
            initial_lane_id (str, optional): Initial lane ID of the vehicle. Defaults to None.
        """
        if conf.simulation_config["speed_mode"] == "low_speed":
            self.v_low, self.v_high = 0, 20
        elif conf.simulation_config["speed_mode"] == "high_speed":
            self.v_low, self.v_high = 20, 40
        self.id = id
        self.controller = controller
        controller.vehicle = self
        self.observation_method = observation_method
        self._recent_observation = None
        self.simulator = simulator
        self.action_step_size = self.simulator.action_step_size
        self.step_size = self.simulator.step_size
        self.lc_duration = self.simulator.lc_duration
        self.lc_step_num = int(self.simulator.lc_duration / self.action_step_size)
        self.routeID = routeID
        self.initial_speed = initial_speed
        self.initial_position = initial_position
        self.initial_lane_id = initial_lane_id
        self.controlled_flag = False
        self.controlled_duration = 0
        self.target_lane_index = None

    def __iter__(self):
        yield self

    @property
    def type(self):
        """Separate BV with AV through the type property.

        Returns:
            str: Type of the vehicle based on its ID. "BV" stands for background vehicle, "AV" stands for autonomous vehicle, "Unknown" stands for unknown type.
        """
        return self.id.split("_")[0]

    # @profile
    def install_controller(self, controller):
        """Install controller for each vehicle and change vehicle mode based on controller type.

        Args:
            controller (Controller): Controller object for the vehicle.
        """        
        self.controller = controller
        controller.vehicle = self
        self.controller.install()

    def __str__(self):
        return f'Vehicle(id: {self.id})'

    def __repr__(self):
        return self.__str__()

    # @profile
    def act(self, action):
        """Vehicle acts based on the input action.

        Args:
            action (dict): Lonitudinal and lateral actions. It should have the format: {'longitudinal': float, 'lateral': str}. The longitudinal action is the longitudinal acceleration, which should be a float. The lateral action should be the lane change direction. 'central' represents no lane change. 'left' represents left lane change, and 'right' represents right lane change.

        Returns:
            float: Desired lateral speed.
            float: Lateral adjustment.
        """ 
        self.simulator.set_vehicle_speedmode(self.id, 0)
        self.simulator.set_vehicle_lanechangemode(self.id, 0)
        controlled_acc = action["longitudinal"]
        current_velocity = self.observation.information["Ego"]["velocity"]
        lateral_adjustment = 0
        
        if action["lateral"] == "central":
            current_lane_offset = self.simulator.get_vehicle_lateral_lane_position(self.id)
            if abs(current_lane_offset) > max_sublane_distance_per_step:
                lateral_adjustment = -math.copysign(1, current_lane_offset)*max_sublane_distance_per_step
                self.simulator.change_vehicle_sublane_dist(self.id, lateral_adjustment, self.step_size)
            else:
                lateral_adjustment = -current_lane_offset
                self.simulator.change_vehicle_sublane_dist(self.id, lateral_adjustment, self.step_size)

            if current_velocity + controlled_acc*self.step_size > self.v_high:
                controlled_acc = (self.v_high - current_velocity)/self.step_size
            elif current_velocity + controlled_acc* self.step_size < self.v_low:
                controlled_acc = (self.v_low - current_velocity)/self.step_size
            self.simulator.change_vehicle_speed(self.id, controlled_acc, self.action_step_size)
            desired_lat_v = 0
        else:
            desired_lat_v = self.simulator.change_vehicle_lane(self.id, action["lateral"], self.lc_duration)
            if current_velocity + controlled_acc*self.lc_duration > self.v_high:
                controlled_acc = (self.v_high - current_velocity)/self.lc_duration
            elif current_velocity + controlled_acc* self.lc_duration < self.v_low:
                controlled_acc = (self.v_low - current_velocity)/self.lc_duration
            self.simulator.change_vehicle_speed(self.id, controlled_acc, self.lc_duration)
        
        return desired_lat_v, lateral_adjustment
        
    def is_action_legal(self, action):
        """Check if the action is legal for the vehicle.

        Args:
            action (dict): Action index for the controlled background vehicle.

        Returns:
            bool: True if the action is legal, otherwise False.
        """
        if self.id not in self.simulator.get_vehID_list():
            return False
        if action["lateral"] == "left" and not self.simulator.get_vehicle_lane_adjacent(self.id, 1):
            return False
        if action["lateral"] == "right" and not self.simulator.get_vehicle_lane_adjacent(self.id, -1):
            return False
        return True

    # @profile
    def update(self):
        """Update the state of the background vehicle, including conducting actions, setting colors and maintain controlled_flag and controlled_duration.
        """
        # print(self.id, self.controller.action, self.controlled_flag, self.simulator.get_vehicle_speed_without_traci(self.id), self.simulator.get_vehicle_speed(self.id))
        # Replay module
        if self.simulator.replay_flag:
            time_step = self.simulator.get_time()
            if time_step in self.simulator.input_trajs.keys():
                time_info = self.simulator.input_trajs[time_step]
                if self.id in time_info.keys():
                    veh_info = time_info[self.id]
                    self.simulator.change_vehicle_speed(self.id, float(veh_info["acceleration"]), self.action_step_size)
                    self.simulator.change_vehicle_sublane_dist(self.id, float(veh_info["y"])-self.observation.information["Ego"]["position"][1], self.action_step_size)
                    # self.simulator.change_vehicle_position(self.id, (float(veh_info['x']), float(veh_info['y'])), angle=float(veh_info['angle']), keepRoute=0)
                else:
                    print("No information for vehicle", self.id, "at SUMO time", time_step)
            if self.id == "CAV" and self.simulator.input_trajs_add is not None:
                if "CAV_info" in self.simulator.input_trajs_add.keys() and str(time_step) in self.simulator.input_trajs_add["CAV_info"].keys() and self.simulator.input_trajs_add["CAV_info"][str(time_step)]["RSS_control_flag"]:
                    self.simulator.set_vehicle_color(self.id, self.color_orange)
                else:
                    self.simulator.set_vehicle_color(self.id, self.color_red)
        elif conf.train_mode == "offline":
            time_step = str(round(self.simulator.get_time()+float(self.simulator.env.init_time_step),1))
            if time_step in self.simulator.env.replay_trajs.keys():
                time_info = self.simulator.env.replay_trajs[time_step]
                if self.id in time_info.keys():
                    veh_info = time_info[self.id]
                    self.simulator.change_vehicle_speed(self.id, float(veh_info["acceleration"]), self.action_step_size)
                    self.simulator.change_vehicle_sublane_dist(self.id, float(veh_info["y"])-self.observation.information["Ego"]["position"][1], self.action_step_size)
            else:
                if self.simulator.env.last_nade_info is not None and self.id in self.simulator.env.last_nade_info.keys():
                    lateral_offset = self.simulator.get_vehicle_lateral_lane_position(self.id)
                    if abs(lateral_offset)> 0.01:
                        self.simulator.change_vehicle_speed(self.id, 0.0, self.action_step_size)
                        self.simulator.change_vehicle_sublane_dist(self.id, (1-2*self.simulator.env.last_nade_info[self.id][0])*0.4, self.action_step_size)
                    else:
                        self.simulator.set_vehicle_speed(self.id, -1)
                else:
                    self.simulator.set_vehicle_speed(self.id, -1)
        # Normal module
        elif self.controller.action is not None and not self.controlled_flag and self.is_action_legal(self.controller.action):
            # Control the vehicle for the first time.
            self.act(self.controller.action)
            # print(self.simulator.getLateralLanePosition(self.id))
            if self.controller.action["lateral"] == "left" or self.controller.action["lateral"] == "right":
                self.controlled_duration += 1
            self.controlled_flag = True
            
    def reset_control_state(self):
        """Reset the control state of the vehicle, including setting the controlled_flag to be false, and setting the vehicle color.
        """        
        if self.controlled_flag:
            if self.controlled_duration == 0:
                self.controller.action = None
                self.controlled_flag = False
                self.controller.reset()
            else:
                self.controlled_duration = (self.controlled_duration + 1)%self.lc_step_num
 
    @property
    def observation(self):
        """Observation of the vehicle.

        Returns:
            Observation: Information of the vehicle itself and its surroundings. 
        """        
        if not self._recent_observation or self._recent_observation.time_stamp != self.simulator.get_time(): #if the recent observation exists and the recent observation is not updated at the current timestamp, update the recent observation
            self._recent_observation = self._get_observation()
        return self._recent_observation

    # @profile
    def _get_observation(self):
        """Get observation of the vehicle at the last time step. Do not directly use this method, instead, use the observation() property method for efficient performance.

        Returns:
            Observation: Surrounding information the vehicle along with the information of itself, including vehicle ID, speed, position, lane index, and vehicle distance.
        """
        obs = self.observation_method(ego_id=self.id, time_stamp=self.simulator.get_time())
        obs.update(self.simulator.get_vehicle_context_subscription_results(self.id), self.simulator, self)
        return obs
    

class VehicleDynamics(Vehicle):
    def __init__(self, id, controller, observation_method=Observation, routeID=None, simulator=None, initial_speed=None, initial_position=None, initial_lane_id=None):
        """Initialize the vehicle dynamics object.

        Args:
            id (str): Vehicle ID.
            controller (Controller): Controller object for the vehicle.
            observation_method (Observation, optional): Observation method for the vehicle. Defaults to Observation.
            routeID (str, optional): Route ID of the vehicle. Defaults to None.
            simulator (Simulator, optional): Simulator object for the vehicle. Defaults to None.
            initial_speed (float, optional): Initial speed of the vehicle. Defaults to None.
            initial_position (list, optional): Initial position of the vehicle. Defaults to None.
            initial_lane_id (str, optional): Initial lane ID of the vehicle. Defaults to None.
        """
        super().__init__(id, controller, observation_method=observation_method, routeID=routeID, simulator=simulator, initial_speed=initial_speed, initial_position=initial_position, initial_lane_id=initial_lane_id)
        self.params = {
            "L": 2.54, # wheel base (m)
            "a": 1.14, # distance c.g. to front axle (m)
            "m": 1500., # mass (kg)
            "Iz": 2420., # yaw moment of inertia (kg-m^2)
            "Caf": 44000.*2, # cornering stiffness -- front axle (N/rad)
            "Car": 47000.*2, # cornering stiffness -- rear axle (N/rad)
            "g": 9.81
        }
        lane_center_y = [42., 46., 50.]
        if not initial_speed:
            initial_speed = 30.0
        if not initial_lane_id:
            self.initial_pos = [400.,46.]
        else:
            initial_lane_index = int(initial_lane_id.split("_")[-1])
            self.initial_pos = [400.,lane_center_y[initial_lane_index]]
        self.states = [initial_speed,0,0,np.pi/2,0,0]
        self.state_list = [self.states]
        self.v_list = [initial_speed]
        self.t = 0
        self.veh_length = 5
        self.veh_width = 1.8
        self.road_width = 4.0
        self.lateral_speed = 0.0

    def act(self, action):
        """Act based on the input action.

        Args:
            action (dict): Action index for the controlled background vehicle.

        Raises:
            NotImplementedError: When the action is not excutable, raise error.
        """
        if "acceleration" in action and "steering_angle" in action:
            self.act_realcontrol(action)
        elif "lateral" in action and "longitudinal" in action:
            self.act_simulation(action)
        else:
            raise NotImplementedError(f"Action: {action}, this kind of action is not excutable!")
    
    def act_realcontrol(self, action):
        """Act based on the input action using the sophisticated vehicle dynamics.

        Args:
            action (dict): Action information for the controlled vehicle.
        """
        # check action limit
        anglemax = 10
        action["acceleration"] = np.clip(action["acceleration"], conf.acc_low, conf.acc_high)
        action["steering_angle"] = np.clip(action["steering_angle"], -anglemax, anglemax)
        if conf.speedlimit_check_flag:
            # check speed limit
            pred_long_v = self.states[0] + action["acceleration"] * self.step_size
            if pred_long_v > self.v_high:
                action["acceleration"] = (self.v_high-self.states[0])/self.step_size
            elif pred_long_v < self.v_low:
                action["acceleration"] = (self.v_low-self.states[0])/self.step_size
        # print(self.simulator.get_vehicle_roadID(self.id), self.simulator.get_vehicle_lane_position(self.id))
        new_states = update_vehicle_real_states(self.states, action, self.params, self.step_size)
        dx = new_states[1] - self.states[1]
        current_v = self.observation.information["Ego"]["velocity"]
        new_v = new_states[0]*np.cos(new_states[3]-lane_angle)-new_states[4]*np.sin(new_states[3]-lane_angle)
        # if new_v > conf.v_high or new_v < conf.v_low:
        #     print(current_v, new_v, action, self.states, new_states)
        acc = (new_v-current_v)/self.step_size
        # acc = action["acceleration"]
        # if acc < -4 or acc > 2:
        #     cur_v_ins = self.states[0]*np.cos(self.states[3]-np.pi/2)-self.states[4]*np.sin(self.states[3]-np.pi/2)
        #     next_v_ins = new_states[0]*np.cos(new_states[3]-np.pi/2)-new_states[4]*np.sin(new_states[3]-np.pi/2)
        #     print(current_v, new_v, self.states, new_states)
        # self.simulator.change_vehicle_speed(self.id, acc, self.step_size)
        # self.simulator.change_vehicle_sublane_dist(self.id, dy, self.step_size)
        current_pos = self.observation.information["Ego"]["position"]
        current_heading = self.observation.information["Ego"]["heading"]
        dy = new_states[2] - self.states[2]
        new_pos = [current_pos[0]-self.veh_length/2*np.sin(current_heading/180*np.pi)+dx, current_pos[1]-self.veh_length/2*np.cos(current_heading/180*np.pi)+dy]
        within_flag = True
        if conf.road_check_flag:
            road_constraints = [[-np.inf,np.inf], [40+self.road_width/2, 52-self.road_width/2]]
            
            # add hard constraint for road geometry
            new_pos, new_states, within_flag = check_network_boundary(new_pos, road_constraints, new_states, self.initial_pos)
        if not within_flag:
            new_lateral_v = 0.
            new_states[0] = new_v # maintain vehicle longitudinal speed
        else:
            dy = new_states[2] - self.states[2]
            new_lateral_v = -new_states[0]*np.sin(new_states[3]-lane_angle)-new_states[4]*np.cos(new_states[3]-lane_angle)
        new_front_pos = [new_pos[0]+self.veh_length/2*np.sin(new_states[3]), new_pos[1]+self.veh_length/2*np.cos(new_states[3])]
        self.simulator.change_vehicle_position(self.id, new_front_pos, angle=new_states[3]/np.pi*180, keepRoute=0)
        self.simulator.set_vehicle_speed(self.id, new_v)
        self.states = list(new_states)
        self.lateral_speed = new_lateral_v
        self.state_list.append(new_states)
        self.v_list.append(new_v)
        self.controlled_duration = 0

    def act_simulation(self, action):
        """Act based on the input action using the simplified vehicle dynamics.

        Args:
            action (dict): Action information for the controlled vehicle.
        """
        desired_lat_v,lateral_adjustment = 0,0
        if self.is_action_legal(action) and not self.controlled_flag: # currently not do lane change
            desired_lat_v,lateral_adjustment = super().act(action)

        deltat = self.step_size
        lon_v = self.observation.information["Ego"]["velocity"]
        new_lon_v = lon_v+deltat*action["longitudinal"]
        new_lat_v = lat_v = self.observation.information["Ego"]["lateral_speed"]
        dx,dy = 0,lateral_adjustment
        if not self.controlled_flag and action["lateral"] != "central": 
            new_lat_v = desired_lat_v * (2*int(action["lateral"]=="left")-1) # for time step deciding to do the lane change
            self.controlled_duration = 1
        elif self.controlled_flag and self.controlled_duration <= self.lc_step_num-1: # for time step doing the lane change
            if self.controlled_duration == 0: # for thhe last time step of lane change
                new_lat_v = 0
                dy += lat_v*deltat
        dx += lon_v*deltat+0.5*action["longitudinal"]*deltat**2
        dy += new_lat_v*deltat
        new_states = [
            np.sqrt(new_lon_v**2+new_lat_v**2),
            self.states[1]+dx,
            self.states[2]+dy,
            math.radians(self.observation.information["Ego"]["heading"]),
            0,
            0
        ]
        self.states = list(new_states)
        self.lateral_speed = new_lat_v
        self.state_list.append(new_states)

    def update(self):
        """Update the state of the background vehicle, including conducting actions, setting colors and maintain controlled_flag and controlled_duration.
        """
        if not self.simulator.replay_flag and conf.train_mode != "offline":
            self.act(self.controller.action)
        elif conf.train_mode == "offline":
            self.act(self.controller.action)
        else:
            time_step = self.simulator.sumo_time_stamp
            if time_step==10.0:
                print("find")
            if self.id == "CAV" and self.simulator.input_trajs_add is not None:
                if "CAV_info" in self.simulator.input_trajs_add.keys() and str(time_step) in self.simulator.input_trajs_add["CAV_info"].keys():
                    action = self.simulator.input_trajs_add["CAV_info"][str(time_step)]["CAV_action"]
                    self.act(action)
                    time_info = self.simulator.input_trajs[self.simulator.get_time()]
                    veh_info = time_info[self.id]
                    self.simulator.change_vehicle_position(self.id, (float(veh_info['x']), float(veh_info['y'])), angle=float(veh_info['angle']), keepRoute=0)
                    self.simulator.set_vehicle_speed(self.id, float(veh_info['speed']))
                if "CAV_info" in self.simulator.input_trajs_add.keys() and str(time_step) in self.simulator.input_trajs_add["CAV_info"].keys() and self.simulator.input_trajs_add["CAV_info"][str(time_step)]["NADE_critical"]:
                    self.simulator.set_vehicle_color(self.id, self.color_purple)
                else:
                    self.simulator.set_vehicle_color(self.id, self.color_red)
        self.t += self.step_size
        self.controlled_flag = True

    def reset_control_state(self):
        """Reset the control state of the vehicle, including setting the controlled_flag to be false, and setting the vehicle color.
        """
        if self.controlled_flag:
            if self.controlled_duration == 0:
                self.controller.action = None
                self.controlled_flag = False
                self.controller.reset()
            else:
                self.controlled_duration = (int(self.controlled_duration) + 1)%self.lc_step_num
    
    def update_dynamics(self, parameters):
        """Update vehicle dynamics

        Args:
            parameters (dict): Important vehicle dynamics parameters. 

        Raises:
            ValueError: When the important vehicle dynamics parameters are not included, raise error.
        """
        for param in ["L", "a", "m", "Iz", "Caf", "Car", "g"]:
            if param not in parameters.keys():
                raise ValueError("Important vehicle paramrter "+param+" is not included in the expected vehicle dynamics, please correct it!")
        self.params = parameters


class VehicleList(dict):
    def __init__(self, d):
        """A vehicle list that store vehicles. It derives from a dictionary so that one can call a certain vehicle in O(1) time. Rewrote the iter method so it can iterate as a list.
        """
        super().__init__(d)

    def __add__(self, another_vehicle_list):
        if not isinstance(another_vehicle_list, VehicleList):
            raise TypeError('VehicleList object can only be added to another VehicleList')
        vehicle_list = copy(self)
        keys = self.keys()
        for v in another_vehicle_list:
            if v.id in keys:
                print(f'WARNING: vehicle with same id {v.id} is added and overwrote the vehicle list')
            vehicle_list[v.id] = v
        return vehicle_list

    def add_vehicles(self, vlist):
        """Add vehicles to the vehicle list.

        Args:
            vlist (list(Vehicle)): List of Vehicle object or a single Vehicle object.
        """        
        for v in vlist:
            if v.id in self.keys():
                # print(f'WARNING: vehicle with same id {v.id} exists and this vehicle is dumped and not overriding the vehicle with same id in the original list')
                continue
            self[v.id] = v    

    def __iter__(self):
        for k in self.keys():
            yield self[k]