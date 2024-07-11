from math import exp, isclose
from mtlsp.logger.infoextractor import InfoExtractor
import copy, json, os
import numpy as np
from pathlib import Path
from functools import reduce

if os.environ["mode"] == "testing":
    import conf.conf_testing as conf
elif os.environ["mode"] == "training":
    import conf.conf_training as conf
else:
    raise ValueError("Please set the mode to testing or training")
import shutil
import xmltodict


class NADEInfoExtractor(InfoExtractor):  
    def __init__(self, env):
        super().__init__(env)
        self.episode_log = {
            "episode_info": None,
            "collision_result": None,
            "collision_id": None,
            "weight_episode": 1,
            "current_weight": 1,
            "weight_step_info": {},
            "weight_list_step": {},
            "crash_decision_info": None,
            "decision_time_info": {},
            "drl_epsilon_step_info": {},
            "real_epsilon_step_info": {},
            "criticality_step_info": {},
            "drl_obs_step_info": {},
            "ndd_step_info": {},
            "cav_mean_speed": 0,
        }
        self.record = {}
        self.initial_log = {}
        self.cav_speed_list = []
        self.cav_mean_speed = 0
        self.cav_info = {}
        self.RSS_step_num, self.total_step_num = 0, 0
        self.reason = {}
        self.log_safe_flag = False

    def traj_to_info(self, prev_traj):
        """Convert the trajectory information to JSON format.

        Args:
            prev_traj (Dict): Trajectory information.

        Returns:
            Dict: Serialized trajectory information.
        """        
        if prev_traj is None:
            return None
        json_traj = {}
        for vehicle in prev_traj:
            json_traj[vehicle] = {}
            for action in prev_traj[vehicle]:
                json_traj[vehicle][action] = prev_traj[vehicle][action].traj_info
        return json_traj

    def add_initialization_info(self, vehID, information):
        """Add information about the initialization of the episode.

        Args:
            vehID (str): Vehicle ID.
            information (dict): Vehicle information including speed, postion, and lane index.
        """
        self.initial_log[vehID] = copy.deepcopy(information)

    # @profile
    def get_terminate_info(self, stop, reason, additional_info):
        """Obtain the information of the termination reason and all the vehicles in the simulation.

        Args:
            stop (bool): Whether the simulation is stopped.
            reason (dict): Stop reason flag.
            additional_info (dict): Collision vehicle ID.
        """
        if conf.experiment_config["mode"] == "DRL_train":
            return None
        if stop:
            self.reason = reason
            self.episode_log["episode_info"] = self.env.episode_info
            self.cav_mean_speed = np.mean(self.cav_speed_list)
            self.episode_log["cav_mean_speed"] = self.cav_mean_speed
            self.episode_log["RSS_rate"] = [self.RSS_step_num, self.total_step_num]

            if 1 in self.reason:  # have crash
                self.episode_log["collision_result"] = 1
                self.episode_log["collision_id"] = additional_info["collision_id"]
                crash_decision_info = {}
                all_vehicle_list = self.env.vehicle_list
                for id in self.episode_log["collision_id"]:
                    crash_decision_info[id] = all_vehicle_list[id].controller.ego_info
                    crash_decision_info[id]["action"] = all_vehicle_list[
                        id
                    ].controller.action
                self.episode_log["crash_decision_info"] = crash_decision_info
                if not conf.nade_offline_collection:
                    self.episode_log["CAV_info"] = self.cav_info
                    self.episode_log["NADE_info"] = (
                        self.env.global_controller_instance_list[0].nade_info
                    )
            else:
                self.episode_log["collision_result"] = 0
                if (
                    conf.more_info_critical
                    or conf.experiment_config["log_mode"] == "offlinecollect"
                    or conf.experiment_config["log_mode"] == "offlinecollect_all"
                    or (
                        self.env.simulator.input is not None
                        and self.env.simulator.replay_flag
                    )
                ):
                    self.episode_log["CAV_info"] = self.cav_info
                    self.episode_log["NADE_info"] = (
                        self.env.global_controller_instance_list[0].nade_info
                    )

            crash_log_flag = 1 in reason
            all_log_flag = conf.experiment_config["log_mode"] == "all"
            self.episode_log["initial_criticality"] = 1
            self.episode_log["initial_weight"] = self.env.initial_weight
            self.episode_log["reject_flag"] = False
            if not (
                crash_log_flag or all_log_flag
            ):  # if log all or crash event happens
                # if 1:
                self.episode_log["decision_time_info"] = (
                    None  # will delete the most heavy part, only log the overall info
                )
                self.episode_log["crash_decision_info"] = None
            elif self.episode_log["weight_episode"] < 0.9:
                self.episode_log["decision_time_info"] = None
                self.episode_log["crash_decision_info"] = None
            # self.delete_unused_epsilon_and_ndd_info()
            # self.delete_unused_cav_info()
            save_dir = os.path.join(self.save_dir, "rejected")
            # print("Reason:", reason)
            if 1 in self.reason:
                # print("CRASH!!!!!!!")
                save_dir = os.path.join(self.save_dir, "crash")
            if 4 in self.reason or 9 in self.reason:
                save_dir = os.path.join(self.save_dir, "tested_and_safe")
            if 6 in self.reason:
                save_dir = os.path.join(self.save_dir, "leave_network")
            if 7 in self.reason or 8 in self.reason:
                save_dir = os.path.join(self.save_dir, "speed_out_of_range")
            # with open(save_dir + "/"+str(self.episode_log["episode_info"]["id"]) + ".json", 'w') as json_file:
            #     json_file.write(json_str)
            if 1 in self.reason:
                # if 1:
                save_dir = os.path.join(self.save_dir, "crash")
                if not conf.compress_log_flag:
                    with open(
                        save_dir
                        + "/"
                        + str(self.episode_log["episode_info"]["id"])
                        + ".json",
                        "w",
                    ) as json_file:
                        json_str = json.dumps(self.episode_log, indent=4)
                        json_file.write(json_str)
                else:
                    with open(
                        save_dir + "/" + str(self.env.simulator.worker_id) + ".json",
                        "a",
                    ) as json_file:
                        json.dump(self.episode_log, json_file)
                        json_file.write("\n")
                self.weight_result = float(self.episode_log["weight_episode"])
                # print("CRASH WEIGHT RESULT:", self.weight_result)
            else:
                # save_name = save_dir + "/"+str(self.episode_log["episode_info"]["id"]) + ".json"
                self.log_safe_flag = False  # False
                if (
                    conf.more_info_critical
                    and self.cav_info != {}
                    and "NADE_info" in self.episode_log.keys()
                ):
                    for time_index in self.episode_log["NADE_info"].keys():
                        nade_dict = self.episode_log["NADE_info"][time_index]
                        for action in nade_dict.values():
                            if action in [0, 1]:
                                self.log_safe_flag = True
                        if self.log_safe_flag:
                            break
                elif conf.experiment_config["log_mode"] == "offlinecollect":
                    highly_crit_list = [
                        self.episode_log["CAV_info"][step]["NADE_critical"]
                        for step in self.episode_log["CAV_info"]
                    ]
                    if (
                        self.episode_log["weight_episode"] < 1
                        or self.env.three_circle_min_distance < 4
                    ):
                        self.log_safe_flag = sum(highly_crit_list) > 0
                elif conf.experiment_config["log_mode"] == "offlinecollect_all":
                    self.log_safe_flag = True
                elif (
                    self.env.simulator.input is not None
                    and self.env.simulator.replay_flag
                ):
                    self.log_safe_flag = True
                if self.log_safe_flag:
                    save_dir = os.path.join(self.save_dir, "tested_and_safe")
                    if not conf.compress_log_flag:
                        with open(
                            save_dir
                            + "/"
                            + str(self.episode_log["episode_info"]["id"])
                            + ".json",
                            "w",
                        ) as json_file:
                            json_str = json.dumps(self.episode_log, indent=4)
                            json_file.write(json_str)
                    else:
                        with open(
                            save_dir
                            + "/"
                            + str(self.env.simulator.worker_id)
                            + ".json",
                            "a",
                        ) as json_file:
                            json.dump(self.episode_log, json_file)
                            json_file.write("\n")
                # else:
                #     Path(save_name).touch(exist_ok=True)
                # with open(save_dir + "/"+str(self.episode_log["episode_info"]["id"]) + ".json", 'w') as json_file:
                #     json_file.write(json_str)
                self.weight_result = 0

            self.episode_log = {
                "collision_result": None,
                "collision_id": None,
                "weight_episode": 1,
                "current_weight": 1,
                "episode_info": None,
                "crash_decision_info": None,
                "decision_time_info": {},
                "weight_step_info": {},
                "drl_epsilon_step_info": {},
                "real_epsilon_step_info": {},
                "weight_list_step": {},
                "criticality_step_info": {},
                "drl_obs_step_info": {},
                "ndd_step_info": {},
                "cav_mean_speed": 0,
            }
            self.cav_info = {}
            self.cav_speed_list = []
            self.RSS_step_num, self.total_step_num = 0, 0

    def get_nadecriticality_this_step(self):
        """Obtain the criticality of the current step.

        Returns:
            float: Criticality of the current step.
        """
        return self.env.global_controller_instance_list[0].control_log["criticality"]

    def get_current_drl_obs(self):
        """Obtain the observation of the current step for Intelligent Testing Environment.

        Returns:
            np.array: Observation of the current step.
        """
        return self.env.global_controller_instance_list[0].control_log[
            "discriminator_input"
        ]

    def delete_unused_epsilon_and_ndd_info(self):
        """Remove the unused epsilon and NDD information.
        """
        # weight_timestep = self.episode_log["weight_step_info"].keys()
        # drl_epsilon_step_info = self.episode_log["drl_epsilon_step_info"].keys()
        # real_epsilon_step_info = self.episode_log["real_epsilon_step_info"].keys()
        # ndd_step_info_keys = self.episode_log["ndd_step_info"].keys()
        for key in list(self.episode_log["drl_epsilon_step_info"].keys()):
            if key not in self.episode_log["weight_step_info"]:
                self.episode_log["drl_epsilon_step_info"].pop(key)
        for key in list(self.episode_log["real_epsilon_step_info"].keys()):
            if key not in self.episode_log["weight_step_info"]:
                self.episode_log["real_epsilon_step_info"].pop(key)
        for key in list(self.episode_log["ndd_step_info"].keys()):
            if key not in self.episode_log["weight_step_info"]:
                self.episode_log["ndd_step_info"].pop(key)

    def delete_unused_cav_info(self):
        """Remove the unused CAV information."""
        weight_threshold = 1.0
        weight_list = np.array(list(self.episode_log["weight_list_step"].values()))
        cum_weight_list = np.cumprod(weight_list)
        for i in range(len(cum_weight_list)):
            if cum_weight_list[i] < weight_threshold:
                break
            else:
                time_step = list(self.episode_log["weight_list_step"].keys())[i]
                self.episode_log["CAV_info"].pop(time_step)

    # @profile
    def get_snapshot_info(self, control_info=None):
        """Obtain the vehicle information at every time step.
        """
        if (
            conf.experiment_config["mode"] == "DRL_train"
            and conf.train_mode == "offline"
        ):  # and not self.env.eval_flag:
            return None
        cav = self.env.vehicle_list["CAV"]
        cav_speed = cav.observation.information["Ego"]["velocity"]
        self.cav_speed_list.append(cav_speed)
        self.save_dir = self.env.simulator.experiment_path
        # ! still need to check the simulation has crash?
        # If crash, then set collision result to 1 and log collision id
        # If the simulation ended, then set collision result to 0
        time_step = self.env.simulator.sumo_time_stamp
        snapshot_weight_list = self.env.global_controller_instance_list[0].control_log[
            "weight_list_per_simulation"
        ]
        self.episode_log["weight_list_step"][time_step] = snapshot_weight_list
        self.episode_log["weight_episode"] = self.episode_log[
            "weight_episode"
        ] * reduce(lambda x, y: x * y, snapshot_weight_list)
        self.episode_log["current_weight"] = reduce(
            lambda x, y: x * y, snapshot_weight_list
        )
        if self.get_nadecriticality_this_step() > 0:
            # if not isclose(self.episode_log["current_weight"], 1):
            self.episode_log["weight_step_info"][time_step] = self.episode_log[
                "current_weight"
            ]
            self.episode_log["criticality_step_info"][
                time_step
            ] = self.get_nadecriticality_this_step()
            self.episode_log["drl_obs_step_info"][
                time_step
            ] = self.get_current_drl_obs()
        if self.env.global_controller_instance_list[0].drl_epsilon_value != -1:
            self.episode_log["drl_epsilon_step_info"][time_step] = (
                self.env.global_controller_instance_list[0].drl_epsilon_value
            )
        if self.env.global_controller_instance_list[0].real_epsilon_value != -1:
            self.episode_log["real_epsilon_step_info"][time_step] = (
                self.env.global_controller_instance_list[0].real_epsilon_value
            )

        if "ndd_possi" in self.env.global_controller_instance_list[0].control_log:
            self.episode_log["ndd_step_info"][time_step] = (
                self.env.global_controller_instance_list[0].control_log["ndd_possi"]
            )
        all_vehicle_list = self.env.vehicle_list
        log_vehicle_candidate_ids = ["CAV"] + [
            vehicle.id
            for vehicle in self.env.global_controller_instance_list[0].nade_candidates
        ]
        desicion_time_info_step = {}
        desicion_time_info_step["predicted_full_obs"] = (
            self.env.global_controller_instance_list[0].predicted_full_obs
        )
        desicion_time_info_step["predicted_traj_obs"] = self.traj_to_info(
            self.env.global_controller_instance_list[0].predicted_traj_obs
        )
        for veh_id in log_vehicle_candidate_ids:
            desicion_time_info_step[veh_id] = all_vehicle_list[
                veh_id
            ].controller.ego_info
            desicion_time_info_step[veh_id]["action"] = all_vehicle_list[
                veh_id
            ].controller.action
            desicion_time_info_step[veh_id]["observation"] = copy.deepcopy(
                all_vehicle_list[veh_id].observation.information
            )
            if veh_id == "CAV":
                desicion_time_info_step[veh_id]["context"] = copy.deepcopy(
                    all_vehicle_list[veh_id].observation.context
                )
                desicion_time_info_step[veh_id]["SM_LC_prob"] = list(
                    self.env.global_controller_instance_list[0].SM_LC_prob
                )
            if veh_id != "CAV":
                desicion_time_info_step[veh_id]["ndd_pdf"] = list(
                    map(float, all_vehicle_list[veh_id].controller.ndd_pdf)
                )
                desicion_time_info_step[veh_id]["critical_pdf"] = list(
                    map(float, all_vehicle_list[veh_id].controller.bv_criticality_array)
                )
                desicion_time_info_step[veh_id]["NADE_flag"] = all_vehicle_list[
                    veh_id
                ].controller.NADE_flag
                desicion_time_info_step[veh_id]["weight"] = all_vehicle_list[
                    veh_id
                ].controller.weight
                desicion_time_info_step[veh_id]["ndd_possi"] = all_vehicle_list[
                    veh_id
                ].controller.ndd_possi
                desicion_time_info_step[veh_id]["critical_possi"] = all_vehicle_list[
                    veh_id
                ].controller.critical_possi
                desicion_time_info_step[veh_id]["epsilon_pdf_array"] = list(
                    all_vehicle_list[veh_id].controller.epsilon_pdf_array
                )
                desicion_time_info_step[veh_id]["bv_challenge_array"] = list(
                    all_vehicle_list[veh_id].controller.bv_challenge_array
                )
        self.episode_log["decision_time_info"][time_step] = desicion_time_info_step

        if not conf.more_info_critical and conf.experiment_config["log_mode"] in [
            "offlinecollect",
            "crash",
            "offlinecollect_all",
        ]:
            self.cav_info[time_step] = {
                "RSS_control_flag": (
                    cav.controller.RSS_model.cav_veh.RSS_control
                    if hasattr(cav.controller, "RSS_model")
                    else None
                ),
                "RSS_control": (
                    cav.controller.RSS_model.cav_veh.action
                    if hasattr(cav.controller, "RSS_model")
                    else None
                ),
                "CAV_action": (
                    cav.controller.action if hasattr(cav.controller, "action") else None
                ),
                "NADE_critical": (
                    cav.controller.highly_critical
                    if hasattr(cav.controller, "highly_critical")
                    else None
                ),
                "slightly_critical": (
                    cav.controller.slightly_critical
                    if hasattr(cav.controller, "slightly_critical")
                    else None
                ),
                "criticality": (
                    cav.controller._recent_critical["criticality"]
                    if hasattr(cav.controller, "_recent_critical")
                    else None
                ),
            }
        elif cav.controller.highly_critical or cav.controller.slightly_critical:
            self.cav_info[time_step] = {
                "RSS_control_flag": cav.controller.RSS_model.cav_veh.RSS_control,
                "RSS_control": cav.controller.RSS_model.cav_veh.action,
                "CAV_action": cav.controller.action,
                "NADE_critical": cav.controller.highly_critical,
                "slightly_critical": cav.controller.slightly_critical,
                "criticality": cav.controller._recent_critical["criticality"],
            }
        self.total_step_num += 1
        if (
            hasattr(cav.controller, "RSS_model")
            and cav.controller.RSS_model.cav_veh.RSS_control
        ):
            self.RSS_step_num += 1

    def handle_fcd_files(self):
        """Combine the trajectory files into one file.
        """
        file = os.path.join(
            self.env.simulator.experiment_path,
            "crash",
            self.env.simulator.output_filename + "." + "fcd.xml",
        )
        if 1 in self.reason:
            save_dir = os.path.join(self.env.simulator.experiment_path, "crash")
            if conf.compress_log_flag:
                if os.path.isfile(file):
                    with open(file) as xml_file:
                        data_dict = xmltodict.parse(xml_file.read())
                        data_dict["original_name"] = self.env.simulator.output_filename
                        with open(
                            save_dir
                            + "/"
                            + str(self.env.simulator.worker_id)
                            + ".fcd.json",
                            "a",
                        ) as json_file:
                            json.dump(data_dict, json_file)
                            json_file.write("\n")
                    os.remove(file)
        elif self.log_safe_flag:
            save_dir = os.path.join(
                self.env.simulator.experiment_path, "tested_and_safe"
            )
            if conf.compress_log_flag:
                if os.path.isfile(file):
                    with open(file) as xml_file:
                        data_dict = xmltodict.parse(xml_file.read())
                        data_dict["original_name"] = self.env.simulator.output_filename
                        with open(
                            save_dir
                            + "/"
                            + str(self.env.simulator.worker_id)
                            + ".fcd.json",
                            "a",
                        ) as json_file:
                            json.dump(data_dict, json_file)
                            json_file.write("\n")
                    os.remove(file)
            else:
                if os.path.isfile(file):
                    shutil.move(
                        file,
                        os.path.join(
                            save_dir,
                            self.env.simulator.output_filename + "." + "fcd.xml",
                        ),
                    )
        else:
            if os.path.isfile(file):
                os.remove(file)
