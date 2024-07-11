from __future__ import division, print_function
from os import stat
import numpy as np
from mtlsp.simulator import Simulator
from mtlsp.controller.vehicle_controller.controller import Controller, BaseController
import os
if os.environ["mode"] == "testing":
    import conf.conf_testing as conf
elif os.environ["mode"] == "training":
    import conf.conf_training as conf
else:
    raise ValueError("Please set the mode to testing or training")
import scipy
import utils

# Lateral policy parameters
CAV_POLITENESS = 0.  # in [0, 1]
CAV_LANE_CHANGE_MIN_ACC_GAIN = 0.1  # [m/s2]
CAV_LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0  # [m/s2]
Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0  # [m/s2]
CAV_LANE_CHANGE_DELAY = 1.0  # [s]
# NDD Vehicle IDM parameters
COMFORT_ACC_MAX = 2  # [m/s2]
COMFORT_ACC_MIN = -4.0  # [m/s2]
DISTANCE_WANTED = 5.0  # [m]
TIME_WANTED = 1.5  # [s]
DESIRED_VELOCITY = 13  # [m/s]
DELTA = 4.0  # []

acc_low = -4
acc_high = 2
LENGTH = 5


def acceleration(ego_vehicle=None, front_vehicle=None, mode=None):
    """Compute an acceleration command with the Intelligent Driver Model. The acceleration is chosen so as to:
        - reach a target velocity;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

    Args:
        ego_vehicle (dict, optional): Information of the vehicle whose desired acceleration is to be computed. It does not have to be an IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to reason about other vehicles behaviors even though they may not IDMs. Defaults to None.
        front_vehicle (dict, optional): Information of the vehicle preceding the ego-vehicle. Defaults to None.
        mode (str, optional): Difference IDM parameters for BV and CAV. Defaults to None.

    Returns:
        float: Acceleration command for the ego-vehicle in m/s^2.
    """
    if not ego_vehicle:
        return 0

    a0 = COMFORT_ACC_MAX
    v0 = DESIRED_VELOCITY
    delt = DELTA
    acceleration = a0 * \
        (1 - np.power(ego_vehicle["velocity"] /
                      v0, delt))
    if front_vehicle is not None:
        r = front_vehicle["distance"]
        d = max(1e-5, r - LENGTH)
        acceleration -= a0 * \
            np.power(desired_gap(ego_vehicle, front_vehicle, mode) / d, 2)
    return acceleration


def desired_gap(ego_vehicle, front_vehicle=None, mode=None):
    """Compute the desired distance between a vehicle and its leading vehicle.

    Args:
        ego_vehicle (dict): Information of the controlled vehicle.
        front_vehicle (dict, optional): Information of the leading vehicle. Defaults to None.
        mode (str, optional): Difference IDM parameters for BV and CAV. Defaults to None.

    Returns:
        float: Desired distance between the two vehicles in m.
    """
    d0 = DISTANCE_WANTED
    tau = TIME_WANTED
    ab = -COMFORT_ACC_MAX * COMFORT_ACC_MIN
    dv = ego_vehicle["velocity"] - front_vehicle["velocity"]
    d_star = d0 + max(0, ego_vehicle["velocity"] * tau +
                      ego_vehicle["velocity"] * dv / (2 * np.sqrt(ab)))
    return d_star


class LowSpeedNDDController(BaseController):
    """A vehicle using both a longitudinal and a lateral decision policies.

        - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and velocity.
        - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    def __init__(self, subscription_method=Simulator.subscribe_vehicle_all_information, controllertype="LowSpeedNDDController"):
        """Initialize the controller.

        Args:
            subscription_method (function, optional): SUMO subscription methods. Defaults to Simulator.subscribe_vehicle_all_information.
            controllertype (str, optional): Type of the controller. Defaults to "LowSpeedNDDController".
        """        
        super().__init__(subscription_method=subscription_method, controllertype=controllertype)
        self.mode = None
        self._recent_ndd_pdf = {"time_step": None, "pdf": None}

    def reset(self):
        """Reset the controller, especially the color of the vehicle, and certain variables. 
        """        
        self.vehicle.simulator.set_vehicle_color(
            self.vehicle.id, self.vehicle.color_green)
        self.NDD_flag, self.NADE_flag = True, False

    def install(self):
        """Install the controller, and set certain parameters for the vehicle.
        """
        super().install()
        self.vehicle.simulator.set_vehicle_color(
            self.vehicle.id, self.vehicle.color_green)
        self.vehicle.simulator.set_vehicle_max_lateralspeed(self.vehicle.id, 4)
        self.NDD_flag, self.NADE_flag = True, False

    @property
    def ndd_pdf(self):
        """Calculate the NDD probability distribution function (pdf) for the vehicle.

        Returns:
            np.array: NDD pdf.
        """        
        if self._recent_ndd_pdf["time_step"] != self.vehicle.simulator.get_time():
            self._recent_ndd_pdf = self.get_ndd_pdf(
                obs=self.vehicle.observation.information)
        return self._recent_ndd_pdf["pdf"]

    # @profile
    def get_ndd_pdf(self, obs=None, external_use=False):
        """Obtain the NDD pdf for the vehicle.

        Args:
            obs (dict, optional): Observation of the vehicle. Defaults to None.
            external_use (bool, optional): A flag indicating if the method is used externally. Defaults to False.

        Returns:
            dict: Dictionary containing the NDD pdf.
        """
        _recent_ndd_pdf = {}
        longi_pdf, lateral_pdf, total_pdf = LowSpeedNDDController.static_get_ndd_pdf(
            obs=obs)
        _recent_ndd_pdf["pdf"] = total_pdf
        if not external_use:
            _recent_ndd_pdf["time_step"] = self.vehicle.simulator.get_time()
        return _recent_ndd_pdf

    @staticmethod
    # @profile
    def static_get_ndd_pdf(obs=None):
        """Obtain the NDD pdf for the vehicle.

        Args:   
            obs (dict, optional): Observation of the vehicle. Defaults to None.
        
        Returns:
            np.array: NDD pdf for the longitudinal maneuvers.
            np.array: NDD pdf for the lateral maneuvers.
            np.array: NDD pdf for the total maneuvers.
        """
        lateral_pdf = LowSpeedNDDController.get_MOBIL_stochastic_pdf(obs)
        longi_pdf = LowSpeedNDDController.stochastic_IDM(
            obs["Ego"], obs["Lead"])
        longi_pdf, lateral_pdf = np.array(longi_pdf), np.array(lateral_pdf)
        total_pdf = [lateral_pdf[0], lateral_pdf[2]] + \
            list(lateral_pdf[1] * longi_pdf)
        return longi_pdf, lateral_pdf, total_pdf

    @staticmethod
    def _Mobil_surrogate_model(cav_obs, lane_index):
        """Apply the Mobil surrogate model for CAV Lane change to calculate the gain for this lane change maneuver. If it does not have safety issue, then return True, gain; otherwise False, None.

        Args:
            lane_index (integer): Candidate lane for the change.

        Returns:
            bool: Safety flag (whether ADS will crash immediately after doing LC).
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
        r_new_preceding, r_new_following, r_new_preceding_1s = 99999, 99999, 99999
        if new_preceding:
            r_new_preceding = new_preceding["distance"]
            r_new_preceding_1s = r_new_preceding + new_preceding["velocity"] - cav_info["velocity"]
        if new_following:
            r_new_following = new_following["distance"]
        if r_new_preceding <= 0 or r_new_following <= 0 or r_new_preceding_1s <= 0:
            return False, gain

        new_following_a = acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = acceleration(
            ego_vehicle=new_following, front_vehicle=cav_info)

        old_preceding = cav_obs["Lead"]
        old_following = cav_obs["Foll"]
        self_pred_a = acceleration(
            ego_vehicle=cav_info, front_vehicle=new_preceding)

        # The deceleration of the new following vehicle after the the LC should not be too big (negative)
        if new_following_pred_a < -conf.Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return True, 0
        if self_pred_a < -conf.Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False, 0

        # Is there an acceleration advantage for me and/or my followers to change lane?
        self_a = acceleration(ego_vehicle=cav_info,
                              front_vehicle=old_preceding)
        old_following_a = acceleration(
            ego_vehicle=old_following, front_vehicle=cav_info)
        old_following_pred_a = acceleration(
            ego_vehicle=old_following, front_vehicle=old_preceding)
        gain = self_pred_a - self_a + conf.Surrogate_POLITENESS * \
            (new_following_pred_a - new_following_a +
             old_following_pred_a - old_following_a)

        # Is there an acceleration advantage for me and/or my followers to change lane?
        self_a = acceleration(ego_vehicle=cav_info, front_vehicle=old_preceding)
        old_following_a = acceleration(ego_vehicle=old_following, front_vehicle=cav_info)
        old_following_pred_a = acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
        gain = self_pred_a - self_a + CAV_POLITENESS * (new_following_pred_a - new_following_a + old_following_pred_a - old_following_a)
        if gain <= CAV_LANE_CHANGE_MIN_ACC_GAIN:
            gain = None
            return False, gain
        

        return True, gain


    @staticmethod
    def get_MOBIL_stochastic_pdf(cav_obs):
        """Obtain the lane change probability of CAV. If ADS will not immediately crash, then the LC probability is at least epsilon_lane_change_prob map gain from [0, 1] to LC probability [epsilon_lane_change_prob, max_remaining_LC_prob].

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
            LC_safety_flag, gain = LowSpeedNDDController._Mobil_surrogate_model(
                cav_obs=cav_obs, lane_index=lane_index)
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
        if not cav_obs["Ego"]["could_drive_adjacent_lane_right"]:
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

    # @profile
    def step(self):
        """Controller decide the next action for the vehicle.

        Returns:
            np.array: NDD pdf.
        """
        super().step()
        final_pdf = self.ndd_pdf
        if self.vehicle.controlled_duration == 0:
            action_id = np.random.choice(
                len(conf.BV_ACTIONS), 1, replace=False, p=final_pdf).item()
            self.action = utils.action_id_to_action_command(action_id)
        return final_pdf

    @staticmethod
    # @profile
    def stochastic_IDM(ego_vehicle, front_vehicle):
        """Calculate the probability distribution function of the acceleration of the vehicle based on the stochastic IDM.

        Args:
            ego_vehicle (dict): Information of the controlled vehicle.
            front_vehicle (dict): Information of the leading vehicle.

        Returns:
            np.array: Probability distribution function of the acceleration of the vehicle.
        """
        tmp_acc = acceleration(ego_vehicle=ego_vehicle,
                               front_vehicle=front_vehicle)
        tmp_acc = np.clip(tmp_acc, conf.acc_low, conf.acc_high)
        acc_possi_list = scipy.stats.norm.pdf(conf.acc_list, tmp_acc, 1)
        # Delete possi if smaller than certain threshold
        acc_possi_list = [
            val if val > conf.Stochastic_IDM_threshold else 0 for val in acc_possi_list]
        assert(sum(acc_possi_list) > 0)
        acc_possi_list = acc_possi_list/(sum(acc_possi_list))

        return acc_possi_list
