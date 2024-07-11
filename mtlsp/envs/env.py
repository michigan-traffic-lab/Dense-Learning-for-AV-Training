from mtlsp.vehicle.vehicle import VehicleList, Vehicle, VehicleDynamics
from abc import abstractmethod
from mtlsp.controller.vehicle_controller.controller import Controller


class BaseEnv(object):
    def __init__(self, global_controller_dict, independent_controller_dict, info_extractor):
        """Initialize the BaseEnv class.

        Args:
            global_controller_dict (dict): Dictionary of global controllers.
            independent_controller_dict (dict): Dictionary of independent controllers.
            info_extractor (object): InfoExtractor object.
        """
        self.episode_info = {"id": 0, "start_time": None, "end_time": None}
        self.vehicle_list = VehicleList({})
        self.departed_vehicle_id_list = []
        self.arrived_vehicle_id_list = []
        self.simulator = None
        self.net = None
        self.global_controller_dict = global_controller_dict
        self.global_controller_instance_list = []
        for veh_type in self.global_controller_dict:
            self.global_controller_instance_list.append(self.global_controller_dict[veh_type](self, veh_type))
        self.independent_controller_dict = independent_controller_dict
        self.info_extractor = info_extractor(self)
    
    def initialize(self):
        """Initialize the environment.
        """
        self.episode_info = {"id": self.simulator.episode, "start_time": self.simulator.get_time(), "end_time": None}
        self.vehicle_list = VehicleList({})
        self.departed_vehicle_id_list = []
        self.arrived_vehicle_id_list = []
        self.global_controller_instance_list = []
        for veh_type in self.global_controller_dict:
            self.global_controller_instance_list.append(self.global_controller_dict[veh_type](self, veh_type))

    def __getattrib__(self, item):
        print(item)
        if item == 'step':
            print('step called')
            self.step()

    def _maintain_all_vehicles(self):
        """Maintain the vehicle list based on the departed vehicle list and arrived vehicle list.
        """        
        self.departed_vehicle_id_list = self.simulator.get_departed_vehID_list()
        self.arrived_vehicle_id_list = self.simulator.get_arrived_vehID_list()
        self._add_vehicles(self.departed_vehicle_id_list)
        self._delete_vehicles(self.arrived_vehicle_id_list)
    
    def _add_vehicles(self, veh_id_list):
        """Add vehicles from veh_id_list.

        Args:
            veh_id_list (list(str)): List of vehicle IDs needed to be inserted.
        """        
        for veh_id in veh_id_list:
            if "CAV" not in veh_id:
                veh = Vehicle(id=veh_id, controller=Controller(), simulator=self.simulator)
            else:
                if not self.simulator.replay_flag or self.simulator.replay_CAV_flag:
                    veh = VehicleDynamics(id=veh_id, controller=Controller(), simulator=self.simulator)
                else:
                    veh = Vehicle(id=veh_id, controller=Controller(), simulator=self.simulator)
            if veh.type in self.independent_controller_dict:
                veh.install_controller(self.independent_controller_dict[veh.type]())
            self.vehicle_list.add_vehicles(veh)

    def _delete_vehicles(self,veh_id_list):
        """Delete vehicles in veh_id_list.

        Args:
            veh_id_list (list(str)): List of vehicle IDs needed to be deleted.
        """        
        for veh_id in veh_id_list:
            self.vehicle_list.pop(veh_id, None)

    def _check_vehicle_list(self):
        """Check the vehicle lists after the simulation step to maintain them again.
        """        
        realtime_vehID_set = set(self.simulator.get_vehID_list())
        vehID_set = set(self.vehicle_list.keys())
        if vehID_set != realtime_vehID_set:
            # print('Warning: the vehicle list is not up-to-date, so update it!')
            for vehID in realtime_vehID_set:
                if vehID not in vehID_set:
                    self._add_vehicles([vehID])
            for vehID in vehID_set:
                if vehID not in realtime_vehID_set:
                    self._delete_vehicles([vehID])
    
    def _replay_vehicle_list(self):
        """Replay the vehicle list based on the input trajectory.
        """
        # Replay module
        realtime_vehID_set = set(self.simulator.get_vehID_list())
        time_stamp = self.simulator.sumo_time_stamp
        if time_stamp in self.simulator.input_trajs.keys():
            vehicle_info = self.simulator.input_trajs[time_stamp]
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
    # @abstractmethod
    def step(self, action=None):
        """Maintain vehicle list and make the simulation step forwards.

        Args:
            action (dict): Action for the controlled vehicles
        """        
        self._maintain_all_vehicles() # maintain both the bvlist and avlist
        self._step(action)

    # @profile       
    @abstractmethod
    def _step(self, action=None):
        """Method that child class MUST implement to specify all actions needed in one step.

        Args:
            action (dict): Action for the controlled vehicles.

        Returns:
            list: List of control information for each vehicle.
        """
        control_info_list = []
        for global_controller in self.global_controller_instance_list:
            control_info = global_controller.step()
            control_info_list.append(control_info)
        self.info_extractor.get_snapshot_info(control_info_list)
        return control_info_list

    # @profile
    def terminate_check(self):
        """Check the termination condition of the simulation.

        Returns:
            bool: Stop flag.
            dict: Reason of the stop.
            dict: Additional information.
        """
        reason, stop, additional_info = self._terminate_check()
        if self.simulator.replay_flag:
            time_step = self.simulator.sumo_time_stamp
            sim_length = max(list(self.simulator.input_trajs.keys()))
            if time_step > sim_length:
                stop = True
        if stop:
            self.episode_info["end_time"] = self.simulator.get_time()-self.simulator.step_size
            try:
                self.info_extractor.get_snapshot_info()
            except:
                pass
            self.info_extractor.get_terminate_info(stop, reason, additional_info)
        return stop, reason, additional_info

    def _terminate_check(self):
        """Method that child class MUST implement to specify the termination condition of the simulation.

        Returns:
            dict: Reason of the stop.
            bool: Stop flag.
            dict: Additional information.
        """
        reason = None
        stop = False
        additional_info = None
        if self.simulator.get_vehicle_min_expected_number() == 0:
            reason = "All Vehicles Left"
            stop = True
        return reason, stop, additional_info