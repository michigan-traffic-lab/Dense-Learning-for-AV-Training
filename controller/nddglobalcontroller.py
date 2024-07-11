from mtlsp.controller.vehicle_controller.globalcontroller import DummyGlobalController
from .nddcontroller import NDDController
from mtlsp.controller.vehicle_controller.controller import Controller

class NDDBVGlobalController(DummyGlobalController):
    def __init__(self, env, veh_type="BV"):
        """Initialize the NDDBVGlobalController class.

        Args:
            env (object): Environment instance.
            veh_type (str): Type of the vehicle.
        """
        super().__init__(env, veh_type)
        self.control_vehicle_set = set()

    # @profile
    def step(self):
        """If there are CAVs in the network, reset the state of all vehicles, update the subscription and decide the next action for each vehicle.
        """        
        if self.apply_control_permission():
            self.reset_control_and_action_state()
            self.update_subscription()
            for veh_id in self.controllable_veh_id_list:
                vehicle = self.env.vehicle_list[veh_id]
                vehicle.controller.step()
                vehicle.update()
        else:
            self.control_vehicle_set = set()
    
    # @profile
    def update_subscription(self, controller=NDDController):
        """Ensure that only the surrounding vehicles of CAV are subscribed. If there are new vehicles, use NDDController to control it and subscribe its information. If one vehicle leaves the observation range of CAV, unsubscribe it and use default SUMO model to control it.

        Args:
            controller (object): Controller instance.
        """        
        sim = self.env.simulator
        CAV = self.env.vehicle_list["CAV"]
        real_veh_list = sim.get_vehID_list()

        if self.env.ndd_local_flag:
            # CAV Local vehicles use NDD model
            context_vehicle_set = set(CAV.observation.context.keys()) 
        else:
            # All vehicles use NDD model
            context_vehicle_set = set(self.env.simulator.get_vehID_list()) - set(["CAV"])
        if self.control_vehicle_set != context_vehicle_set:
            new_control_set = context_vehicle_set
            for veh_id in context_vehicle_set:
                if veh_id not in self.control_vehicle_set and veh_id in self.env.vehicle_list.keys() and veh_id in real_veh_list:
                    sim.subscribe_vehicle_surrounding(veh_id)
                    self.env.vehicle_list[veh_id].install_controller(controller())
                else:
                    new_control_set = new_control_set-{veh_id}
            for veh_id in self.control_vehicle_set:
                if veh_id not in context_vehicle_set and veh_id in self.env.vehicle_list.keys() and veh_id in real_veh_list:
                    sim.unsubscribe_vehicle(veh_id)
                    self.env.vehicle_list[veh_id].install_controller(Controller())
            self.control_vehicle_set = new_control_set
        for veh_id in self.env.vehicle_list.keys()-["CAV"]:
            if veh_id not in context_vehicle_set and self.env.vehicle_list[veh_id].controller.type != "DummyController" and veh_id in real_veh_list:
                sim.unsubscribe_vehicle(veh_id)
                self.env.vehicle_list[veh_id].install_controller(Controller())