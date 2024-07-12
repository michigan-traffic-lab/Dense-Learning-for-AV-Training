from mtlsp.logger.infoextractor import InfoExtractor
import copy, json, os
import numpy as np
from functools import reduce

if os.environ["mode"] == "testing":
    import conf.conf_testing as conf
elif os.environ["mode"] == "training":
    import conf.conf_training as conf
else:
    raise ValueError("Please set the mode to testing or training")


class NDEInfoExtractor(InfoExtractor):
    def __init__(self, env):
        """Initialize the NDEInfoExtractor class.
        
        Args:
            env (object): Environment instance.
        """
        super().__init__(env)
        self.stop_info = {}
        self.episode_log = {}
        self.cav_speed_list = []
        self.cav_mean_speed = 0
        self.cav_info = {}
        self.RSS_step_num, self.total_step_num = 0, 0

    # @profile
    def get_terminate_info(self, stop, reason, additional_info):
        """Obtain the information of the termination reason and all the vehicles in the simulation.

        Args:
            stop (bool): Whether the simulation is stopped.
            reason (dict): Stop reason information.
            additional_info (dict): Additional information, such as the collision vehicle ID.
        """
        if stop:
            self.save_dir = self.env.simulator.experiment_path
            self.episode_log["episode_info"] = self.env.episode_info
            self.stop_info = reason
            self.cav_mean_speed = np.mean(self.cav_speed_list)
            self.episode_log["stop_info"] = reason
            self.episode_log["cav_mean_speed"] = self.cav_mean_speed
            self.episode_log["RSS_rate"] = [self.RSS_step_num, self.total_step_num]
            if 1 in reason:  # have crash
                self.episode_log["collision_result"] = 1
                self.episode_log["collision_id"] = additional_info["collision_id"]
                self.episode_log["CAV_info"] = self.cav_info
            else:
                self.episode_log["collision_result"] = 0
                self.episode_log["collision_id"] = None
            json_str = json.dumps(self.episode_log, indent=4)
            save_dir = os.path.join(self.save_dir, "rejected")
            if 1 in reason:
                save_dir = os.path.join(self.save_dir, "crash")
            if 4 in reason:
                save_dir = os.path.join(self.save_dir, "tested_and_safe")
            if 6 in reason:
                save_dir = os.path.join(self.save_dir, "leave_network")
            if 7 in reason or 8 in reason:
                save_dir = os.path.join(self.save_dir, "speed_out_of_range")
            with open(
                save_dir + "/" + str(self.episode_log["episode_info"]["id"]) + ".json",
                "w",
            ) as json_file:
                json_file.write(json_str)
            if (
                1 not in reason
                and self.env.simulator.experiment_path is not None
                and self.env.simulator.output is not None
                and not (conf.experiment_config["log_mode"] == "all")
            ):
                file = os.path.join(
                    self.env.simulator.experiment_path,
                    "crash",
                    self.env.simulator.output_filename + "." + "fcd.xml",
                )
                if os.path.isfile(file):
                    os.remove(file)
            self.episode_log = {}
            self.cav_info = {}
            self.cav_speed_list = []
            self.RSS_step_num, self.total_step_num = 0, 0

    # @profile
    def get_snapshot_info(self, control_info=None):
        """Obtain the vehicle information at every time step.

        Args:
            control_info (dict): Control information of the vehicle. Default is None.
        """
        cav = self.env.vehicle_list["CAV"]
        cav_speed = cav.observation.information["Ego"]["velocity"]
        self.cav_speed_list.append(cav_speed)
        self.cav_info[self.env.simulator.sumo_time_stamp] = {
            "RSS_control_flag": cav.controller.RSS_model.cav_veh.RSS_control,
            "RSS_control": cav.controller.RSS_model.cav_veh.action,
            "CAV_action": cav.controller.action,
            "NADE_critical": cav.controller.highly_critical,
        }
        self.total_step_num += 1
        if cav.controller.RSS_model.cav_veh.RSS_control:
            self.RSS_step_num += 1
