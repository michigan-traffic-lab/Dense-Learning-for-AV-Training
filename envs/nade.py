import math
from mtlsp.controller.vehicle_controller.idmcontroller import IDMController
from mtlsp.controller.vehicle_controller.globalcontroller import DummyGlobalController
from envs.nde import *
from controller.nadecontroller import NADEBackgroundController
from controller.nadeglobalcontroller import NADEBVGlobalController
from controller.rlcontroller import RLControllerNew
from nadeinfoextractor import NADEInfoExtractor


class NADE(NDE):
    def __init__(self,
        BVController=NADEBackgroundController,
        cav_model="RL"
        ):
        """Initialize the NADE class.

        Args:
            BVController (object): Background vehicle controller.
            cav_model (str): Type of the AV controller.
        """
        if cav_model == "RL":
            cav_controller = AVController
        elif cav_model == "RLNew":
            cav_controller = RLControllerNew
        elif cav_model == "IDM":
            cav_controller = IDMController
        elif cav_model == "IDM_ACM":
            cav_controller = ACMAVIDMController
        elif cav_model == "Surrogate":
            cav_controller = SurrogateIDMAVController
        elif cav_model == "IDM_SafetyGuard":
            cav_controller = IDMController_withSafetyGuard
        elif cav_model == "RL_SafetyGuard":
            cav_controller = RLController_withSafetyGuard
        else:
            raise ValueError(f"Unknown AV controller: {cav_model}!")
        super().__init__(
            AVController=cav_controller,
            BVController=BVController,
            AVGlobalController=DummyGlobalController, 
            BVGlobalController=NADEBVGlobalController,
            info_extractor=NADEInfoExtractor
            )
        self.initial_weight = 1
    
    # @profile
    def _step(self, drl_action=None):
        """NADE subscribes all the departed vehicles and decides how to control the background vehicles.

        Args:
            drl_action (int): Action of the NADE controller.
        """        
        # for vid in self.departed_vehicle_id_list:
        #     self.simulator.subscribe_vehicle(vid)
        self.drl_action = None
        super()._step(action=drl_action)
        self.drl_action = None
