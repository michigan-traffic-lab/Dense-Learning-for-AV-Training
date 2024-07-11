from abc import ABC, abstractmethod
from mtlsp.controller.vehicle_controller.controller import Controller


class GlobalController(ABC):
    """Global Controller, for AV and BV
    """
    def __init__(self, env, veh_type):
        """Initialize the Global Controller class.

        Args:
            env (object): Environment object.
            veh_type (str): Type of the vehicle.
        """
        self.env = env
        self.veh_type = veh_type
        self.control_log = {} # This will have the control log result for each controller
    
    @property
    def controllable_veh_id_list(self):
        """Get the list of controllable vehicle IDs.

        Returns:
            list: List of controllable vehicle IDs.
        """
        return self._get_controllable_veh_id_list()

    def _get_controllable_veh_id_list(self):
        """Helper function to get the list of controllable vehicle IDs.

        Returns:
            list: List of controllable vehicle IDs.
        """
        realtime_vehicle_list = self.env.simulator.get_vehID_list()
        controllable_veh_id_list = []
        for veh in self.env.vehicle_list:
            if veh.type == self.veh_type and veh.id in realtime_vehicle_list:
                controllable_veh_id_list.append(veh.id)
        return controllable_veh_id_list
    
    @abstractmethod
    def step(self):
        pass

    def apply_control_permission(self):
        return True


class DummyGlobalController(GlobalController):
    # @profile
    def reset_control_and_action_state(self):
        """Reset control state of autonomous vehicles.
        """        
        for veh_id in self.controllable_veh_id_list:
            vehicle = self.env.vehicle_list[veh_id]
            vehicle.reset_control_state()
    
    # @profile
    def step(self):
        """Control autonomous vehicles based on controller.
        """
        if self.apply_control_permission():
            self.reset_control_and_action_state()
            for veh_id in self.controllable_veh_id_list:
                vehicle = self.env.vehicle_list[veh_id]
                vehicle.controller.step()
                vehicle.update()

        


