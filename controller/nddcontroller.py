from __future__ import division, print_function
import numpy as np
import bisect
import scipy.io
import os
if os.environ["mode"] == "testing":
    import conf.conf_testing as conf
elif os.environ["mode"] == "training":
    import conf.conf_training as conf
else:
    raise ValueError("Please set the mode to testing or training")
import scipy
import copy
import scipy.stats
import random
import utils
import bisect
from mtlsp.controller.vehicle_controller.controller import Controller, BaseController
from mtlsp.simulator import Simulator


class NDDController(BaseController):

    LENGTH = conf.LENGTH
    speed_lb = conf.v_low
    speed_ub = conf.v_high
    LANE_CHANGE_INDEX_LIST = [0, 1, 2] 

    def __init__(self, controllertype="NDDController"):
        """Initialize the NDDController class.
        """
        super().__init__(subscription_method=Simulator.subscribe_vehicle_all_information,controllertype=controllertype)
        self._recent_ndd_pdf = {"time_step": None, "pdf": None}

    def reset(self):
        """Reset the controller.
        """
        self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_green)
        self.NDD_flag, self.NADE_flag = True, False
    
    # @profile
    def install(self):
        """Install the controller.
        """
        super().install()
        self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_green)
        self.vehicle.simulator.set_vehicle_max_lateralspeed(self.vehicle.id, 4)
        self.NDD_flag, self.NADE_flag = True, False

    @property
    def ndd_pdf(self):
        """Obtain the NDD probability of the BV maneuvers.

        Returns:
            np.array: NDD probability of the BV maneuvers.
        """
        # print(self._recent_ndd_pdf)
        if self._recent_ndd_pdf["time_step"] != self.vehicle.simulator.get_time():
            self._recent_ndd_pdf = self.get_ndd_pdf(obs = self.vehicle.observation.information)
        # a = self.get_ndd_pdf(obs = self.vehicle.observation.information)
        return self._recent_ndd_pdf["pdf"]

    # @profile
    def get_ndd_pdf(self, obs=None, external_use=False):
        """Obtain the NDD probability of the BV maneuvers.

        Args:
            obs (dict): Observation of the surrounding vehicles.
            external_use (bool): Whether the function is used externally.   

        Returns:
            dict: NDD probability of the BV maneuvers.
        """
        _recent_ndd_pdf = {}
        longi_pdf, lateral_pdf, total_pdf = NDDController.static_get_ndd_pdf(obs=obs)
        _recent_ndd_pdf["pdf"] = total_pdf
        if not external_use:
            _recent_ndd_pdf["time_step"] = self.vehicle.simulator.get_time()
        return _recent_ndd_pdf
    
    @staticmethod
    # @profile
    def static_get_ndd_pdf(obs=None):
        """Obtain the NDD probability of the BV maneuvers.

        Args:
            obs (dict): Observation of the surrounding vehicles.

        Returns:
            list: Longitudinal probability of the BV maneuvers.
            list: Lateral probability of the BV maneuvers.
            list: Total probability of the BV maneuvers.
        """
        _, longi_pdf = NDDController.Longitudinal_NDD(obs)
        _, _, lateral_pdf = NDDController.Lateral_NDD(obs)
        if lateral_pdf[0]/conf.lane_change_amplify+lateral_pdf[2]/conf.lane_change_amplify < conf.max_lane_change_probability:
            lateral_pdf = [lateral_pdf[0]/conf.lane_change_amplify, 1-(lateral_pdf[0]+lateral_pdf[2])/conf.lane_change_amplify, lateral_pdf[2]/conf.lane_change_amplify] # divide the lane change probability
        else:
            lateral_pdf = [lateral_pdf[0]/(lateral_pdf[0]+lateral_pdf[2])*conf.max_lane_change_probability, 1-conf.max_lane_change_probability, lateral_pdf[2]/(lateral_pdf[0]+lateral_pdf[2])*conf.max_lane_change_probability]
        total_pdf = [lateral_pdf[0], lateral_pdf[2]] + list(lateral_pdf[1] * longi_pdf)
        return longi_pdf, lateral_pdf, total_pdf

    # @profile
    def step(self):
        """Controller takes one step further.

        Returns:
            np.array: NDD probability of the BV maneuvers.
        """
        super().step()
        final_pdf = self.ndd_pdf
        # if not utils.is_lane_change(self.vehicle.observation.information["Ego"]):
        if self.vehicle.controlled_duration == 0:
            action_id = np.random.choice(len(conf.BV_ACTIONS),1,replace=False,p=final_pdf).item()
            self.action = utils.action_id_to_action_command(action_id)
        return final_pdf
    
    # Round value for lane change
    @staticmethod
    def round_value_lane_change(real_value, value_list, round_item="speed"):
        """Round the value for lane change.

        Args:
            real_value (float): Real value.
            value_list (list): List of values.
            round_item (str): Item to round.

        Returns:
            float: Rounded value.
            int: Index of the rounded value.
        """
        if real_value < value_list[0]: real_value = value_list[0]
        elif real_value > value_list[-1]: real_value = value_list[-1] 

        if conf.round_rule == "Round_to_closest":
            min_val, max_val, resolution = value_list[0], value_list[-1], value_list[1] - value_list[0]
            # real_value_old = np.clip(round((real_value - min_val) / resolution)*resolution + min_val, min_val, max_val)   
            _num = (real_value-min_val)/resolution
            if int(_num*2) == _num*2:
                if int(_num)%2!=0: _num += 0.5
            else:
                _num += 0.5
            real_value_new = int(_num)*resolution + min_val
            # assert real_value_new==real_value_old
            real_value = real_value_new
        
        if round_item == "speed":
            value_idx = bisect.bisect_left(value_list, real_value) 
            value_idx = value_idx if real_value <= value_list[-1] else value_idx - 1
            round_value = value_list[value_idx]
            return round_value, value_idx
        else: 
            value_idx = bisect.bisect_left(value_list, real_value) 
            value_idx = value_idx -1 if real_value != value_list[value_idx] else value_idx
            round_value = value_list[value_idx]
            return round_value, value_idx

    @staticmethod
    # @profile
    def _get_One_lead_LC_prob(veh_front, full_obs):
        """Get the probability of one lead lane change.
        
        Args:
            veh_front (dict): Information of the front vehicle.
            full_obs (dict): Observation of the surrounding vehicles.

        Returns:
            float: Probability of one lead lane change.
            tuple: Related information of the lane change.
        """
        v = full_obs["Ego"]["velocity"]
        if not conf.enable_One_lead_LC:
            return 0, None                   
        r, rr = veh_front["distance"], veh_front["velocity"] - v
        # Check bound
        if not NDDController._check_bound_constraints(v, conf.one_lead_v_low, conf.one_lead_v_high) or not NDDController._check_bound_constraints(r, conf.one_lead_r_low, conf.one_lead_r_high) or not NDDController._check_bound_constraints(rr, conf.one_lead_rr_low, conf.one_lead_rr_high):
            # print("One lead out bound!!!!!!", v, r, rr)
            return 0, None

        round_r, round_r_idx = NDDController.round_value_lane_change(real_value=r, value_list=conf.one_lead_r_list)
        round_rr, round_rr_idx = NDDController.round_value_lane_change(real_value=rr, value_list=conf.one_lead_rr_list)
        round_speed, round_speed_idx = NDDController.round_value_lane_change(real_value=v, value_list=conf.one_lead_speed_list, round_item="speed")
        # Since currently the OL raw data v>=24. So for v<=23, there is definitely no LC, so use the v==24 data when v<=23
        if round_speed <= 23 and conf.OL_LC_low_speed_flag: 
            v_diff = conf.OL_LC_low_speed_use_v-round_speed   
            assert(v_diff > 0)
            round_rr = round_rr - v_diff
            round_rr, round_rr_idx = NDDController.round_value_lane_change(real_value=round_rr, value_list=conf.one_lead_rr_list)
            round_speed, round_speed_idx = NDDController.round_value_lane_change(real_value=conf.OL_LC_low_speed_use_v, value_list=conf.one_lead_speed_list, round_item="speed")
        # assert((round_r-1) <= r and r <= (round_r+1) and (round_rr-1) <= rr and rr <= (round_rr+1))

        lane_change_prob = conf.OL_pdf[round_speed_idx,round_r_idx,round_rr_idx,:][0] * conf.LANE_CHANGE_SCALING_FACTOR
        LC_related = (v, r, rr, round_speed, round_r, round_rr)     

        # chech whether there is LC data in this case
        if conf.OL_pdf[round_speed_idx,round_r_idx,round_rr_idx,:][0] == 0 and conf.OL_pdf[round_speed_idx,round_r_idx,round_rr_idx,:][1] == 0:
            lane_change_prob = None

        return lane_change_prob, LC_related       
    
    @staticmethod
    # @profile
    def _get_Double_LC_prob(veh_adj_front, veh_adj_rear, full_obs): 
        """Get the probability of double lane change.

        Args:
            veh_adj_front (dict): Information of the front vehicle in the adjacent lane.
            veh_adj_rear (dict): Information of the rear vehicle in the adjacent lane.
            full_obs (dict): Observation of the surrounding vehicles.

        Returns:
            float: Probability of double lane change.
            tuple: Related information of the lane change.
        """             
        v = full_obs["Ego"]["velocity"]
        # v_list, r1_list, r2_list, rr1_list, rr2_list = list(np.linspace(conf.lc_v_low,conf.lc_v_high,num=conf.lc_v_num)), list(np.linspace(conf.lc_rf_low,conf.lc_rf_high,num=conf.lc_rf_num)), list(np.linspace(conf.lc_re_low,conf.lc_re_high,num=conf.lc_re_num)), list(np.linspace(conf.lc_rrf_low,conf.lc_rrf_high,num=conf.lc_rrf_num)), list(np.linspace(conf.lc_rre_low,conf.lc_rre_high,num=conf.lc_rre_num))
        v_list, r1_list, r2_list, rr1_list, rr2_list = conf.lc_v_list, conf.lc_rf_list, conf.lc_re_list, conf.lc_rrf_list, conf.lc_rre_list   
        LC_related = None
        # Double lane change
        if not conf.enable_Double_LC:
            return 0, LC_related 
        r1, rr1 = veh_adj_front["distance"], veh_adj_front["velocity"] - v
        r2, rr2 = veh_adj_rear["distance"], v - veh_adj_rear["velocity"]
        if not NDDController._check_bound_constraints(v, conf.lc_v_low, conf.lc_v_high): 
            return 0, LC_related
        elif not NDDController._check_bound_constraints(r1, conf.lc_rf_low, conf.lc_rf_high): 
            return 0, LC_related
        elif not NDDController._check_bound_constraints(rr1, conf.lc_rrf_low, conf.lc_rrf_high): 
            return 0, LC_related             
        elif not NDDController._check_bound_constraints(r2, conf.lc_re_low, conf.lc_re_high): 
            return 0, LC_related
        elif not NDDController._check_bound_constraints(rr2, conf.lc_rre_low, conf.lc_rre_high): 
            return 0, LC_related
        round_v, v_idx = NDDController.round_value_lane_change(real_value=v, value_list=v_list,round_item="speed")
        round_r1, r1_idx = NDDController.round_value_lane_change(real_value=r1, value_list=r1_list)
        round_rr1, rr1_idx = NDDController.round_value_lane_change(real_value=rr1, value_list=rr1_list)
        round_r2, r2_idx = NDDController.round_value_lane_change(real_value=r2, value_list=r2_list)
        round_rr2, rr2_idx = NDDController.round_value_lane_change(real_value=rr2, value_list=rr2_list)
        
        lane_change_prob = conf.DLC_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][0] * conf.LANE_CHANGE_SCALING_FACTOR

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1, round_rr1, round_r2, round_rr2, lane_change_prob)  

        # chech whether there is LC data in this case
        if conf.DLC_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][0] == 0 and conf.DLC_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][1] == 0:
            lane_change_prob = None

        return lane_change_prob, LC_related
    
    @staticmethod
    # @profile
    def _get_Single_LC_prob(veh_front, veh_adj_front, full_obs):       
        """Get the probability of single lane change.

        Args:
            veh_front (dict): Information of the front vehicle.
            veh_adj_front (dict): Information of the front vehicle in the adjacent lane.
            full_obs (dict): Observation of the surrounding vehicles.

        Returns:
            float: Probability of single lane change.
            tuple: Related information of the lane change.
        """       
        v = full_obs["Ego"]["velocity"]
        # v_list, r1_list, r2_list, rr1_list, rr2_list = list(np.linspace(conf.lc_v_low,conf.lc_v_high,num=conf.lc_v_num)), list(np.linspace(conf.lc_rf_low,conf.lc_rf_high,num=conf.lc_rf_num)), list(np.linspace(conf.lc_re_low,conf.lc_re_high,num=conf.lc_re_num)), list(np.linspace(conf.lc_rrf_low,conf.lc_rrf_high,num=conf.lc_rrf_num)), list(np.linspace(conf.lc_rre_low,conf.lc_rre_high,num=conf.lc_rre_num))
        v_list, r1_list, r2_list, rr1_list, rr2_list = conf.lc_v_list, conf.lc_rf_list, conf.lc_re_list, conf.lc_rrf_list, conf.lc_rre_list
        LC_related = None
        # Single lane change
        if not conf.enable_Single_LC:
            return 0, LC_related    
        
        r1, rr1 = veh_front["distance"], veh_front["velocity"] - v
        r2, rr2 = veh_adj_front["distance"], veh_adj_front["velocity"] - v

        if not NDDController._check_bound_constraints(v, conf.lc_v_low, conf.lc_v_high): 
            return 0, LC_related
        elif not NDDController._check_bound_constraints(r1, conf.lc_rf_low, conf.lc_rf_high): 
            return 0, LC_related
        elif not NDDController._check_bound_constraints(rr1, conf.lc_rrf_low, conf.lc_rrf_high): 
            return 0, LC_related             
        elif not NDDController._check_bound_constraints(r2, conf.lc_rf_low, conf.lc_rf_high): 
            return 0, LC_related 
        elif not NDDController._check_bound_constraints(rr2, conf.lc_rrf_low, conf.lc_rrf_high): 
            return 0, LC_related 

        round_v, v_idx = NDDController.round_value_lane_change(real_value=v, value_list=v_list, round_item="speed")
        round_r1, r1_idx = NDDController.round_value_lane_change(real_value=r1, value_list=r1_list)
        round_rr1, rr1_idx = NDDController.round_value_lane_change(real_value=rr1, value_list=rr1_list)
        round_r2, r2_idx = NDDController.round_value_lane_change(real_value=r2, value_list=r2_list)
        round_rr2, rr2_idx = NDDController.round_value_lane_change(real_value=rr2, value_list=rr2_list)        
        
        lane_change_prob = conf.SLC_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][0] * conf.LANE_CHANGE_SCALING_FACTOR

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1, round_rr1, round_r2, round_rr2, lane_change_prob)  

        # chech whether there is LC data in this case
        if conf.SLC_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][0] == 0 and conf.SLC_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][1] == 0:
            lane_change_prob = None

        return lane_change_prob, LC_related

    @staticmethod
    # @profile
    def _get_Cut_in_LC_prob(veh_front, veh_adj_rear, full_obs):
        """Get the probability of cut-in lane change.

        Args:
            veh_front (dict): Information of the front vehicle.
            veh_adj_rear (dict): Information of the rear vehicle in the adjacent lane.
            full_obs (dict): Observation of the surrounding vehicles.

        Returns:
            float: Probability of cut-in lane change.
            tuple: Related information of the lane change.
        """
        v = full_obs["Ego"]["velocity"]
        # v_list, r1_list, r2_list, rr1_list, rr2_list = list(np.linspace(conf.lc_v_low,conf.lc_v_high,num=conf.lc_v_num)), list(np.linspace(conf.lc_rf_low,conf.lc_rf_high,num=conf.lc_rf_num)), list(np.linspace(conf.lc_re_low,conf.lc_re_high,num=conf.lc_re_num)), list(np.linspace(conf.lc_rrf_low,conf.lc_rrf_high,num=conf.lc_rrf_num)), list(np.linspace(conf.lc_rre_low,conf.lc_rre_high,num=conf.lc_rre_num))
        v_list, r1_list, r2_list, rr1_list, rr2_list = conf.lc_v_list, conf.lc_rf_list, conf.lc_re_list, conf.lc_rrf_list, conf.lc_rre_list
        LC_related = None
        
        if not conf.enable_Cut_in_LC:
            return 0, None   

        r1, rr1 = veh_front["distance"], veh_front["velocity"] - v
        r2, rr2 = veh_adj_rear["distance"], v - veh_adj_rear["velocity"]

        if not NDDController._check_bound_constraints(v, conf.lc_v_low, conf.lc_v_high): 
            return 0, LC_related
        elif not NDDController._check_bound_constraints(r1, conf.lc_rf_low, conf.lc_rf_high): 
            return 0, LC_related
        elif not NDDController._check_bound_constraints(rr1, conf.lc_rrf_low, conf.lc_rrf_high): 
            return 0, LC_related             
        elif not NDDController._check_bound_constraints(r2, conf.lc_rf_low, conf.lc_rf_high): 
            return 0, LC_related 
        elif not NDDController._check_bound_constraints(rr2, conf.lc_rrf_low, conf.lc_rrf_high): 
            return 0, LC_related 

        round_v, v_idx = NDDController.round_value_lane_change(real_value=v, value_list=v_list, round_item="speed")
        round_r1, r1_idx = NDDController.round_value_lane_change(real_value=r1, value_list=r1_list)
        round_rr1, rr1_idx = NDDController.round_value_lane_change(real_value=rr1, value_list=rr1_list)
        round_r2, r2_idx = NDDController.round_value_lane_change(real_value=r2, value_list=r2_list)
        round_rr2, rr2_idx = NDDController.round_value_lane_change(real_value=rr2, value_list=rr2_list)        
        
        lane_change_prob = conf.CI_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][0] * conf.LANE_CHANGE_SCALING_FACTOR

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1, round_rr1, round_r2, round_rr2, lane_change_prob) 

        # chech whether there is LC data in this case
        if conf.CI_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][0] == 0 and conf.CI_pdf[v_idx,r1_idx,rr1_idx,r2_idx,rr2_idx,:][1] == 0:
            lane_change_prob = None

        return lane_change_prob, LC_related

    @staticmethod
    # @profile
    def _LC_rear_veh_check(ego_veh, adj_rear_veh):
        """Check the range for ego vehicle and the rear vehicle in the adjacent lane, if it is larger than the threshold, then has no influence.

        Args:
            ego_veh (dict): Information of the ego vehicle.
            adj_rear_veh (dict): Information of the rear vehicle in the adjacent lane.

        Returns:
            bool: Whether the rear vehicle has influence.
            float: Distance between the rear vehicle and the ego vehicle.
        """
        r_adj = adj_rear_veh["distance"]
        if r_adj <= conf.LC_range_threshold:
            return True, r_adj
        else:
            return False, r_adj

    @staticmethod
    # @profile
    def _LC_prob(surrounding_vehicles, full_obs):
        """Get the lane change probability and the expected lane change probability (take the ignored situation into account).

        Args:
            surrounding_vehicles (tuple): Information of the surrounding vehicles.
            full_obs (dict): Observation of all the surrounding vehicles.

        Returns:
            float: Lane change probability.
        """
        LC_prob, E_LC_prob = None, None
        veh_front, veh_adj_front, veh_adj_rear = surrounding_vehicles

        if not veh_adj_front and not veh_adj_rear:
            # One lead LC
            LC_prob, LC_related = NDDController._get_One_lead_LC_prob(veh_front, full_obs)
            E_LC_prob = LC_prob
            return E_LC_prob, "One_lead", LC_related
        
        elif veh_adj_front and not veh_adj_rear:
            # Single lane change
            LC_prob, LC_related = NDDController._get_Single_LC_prob(veh_front, veh_adj_front, full_obs)
            E_LC_prob = LC_prob
            return E_LC_prob, "SLC", LC_related

        elif not veh_adj_front and veh_adj_rear:
            # OL prob
            OL_LC_prob, OL_LC_related = NDDController._get_One_lead_LC_prob(veh_front, full_obs)
                   
            # CI prob
            CI_LC_prob, CI_LC_related = NDDController._get_Cut_in_LC_prob(veh_front, veh_adj_rear, full_obs)
            LC_related = CI_LC_related

            r_adj = veh_adj_rear["distance"]

            if (r_adj >= conf.min_r_ignore) and (CI_LC_prob is not None) and (OL_LC_prob is not None):
                E_LC_prob = conf.ignore_adj_veh_prob * OL_LC_prob + (1-conf.ignore_adj_veh_prob) * CI_LC_prob
            else:
                E_LC_prob = CI_LC_prob
            return E_LC_prob, "Cut_in", LC_related

        elif veh_adj_front and veh_adj_rear:
            # SLC prob
            SLC_LC_prob, SLC_LC_related = NDDController._get_Single_LC_prob(veh_front, veh_adj_front, full_obs)
            
            # DLC prob
            DLC_LC_prob, DLC_LC_related = NDDController._get_Double_LC_prob(veh_adj_front, veh_adj_rear, full_obs)
            LC_related = DLC_LC_related

            r_adj = veh_adj_rear["distance"]

            if (r_adj >= conf.min_r_ignore) and (DLC_LC_prob is not None) and (SLC_LC_prob is not None):
                E_LC_prob = conf.ignore_adj_veh_prob * SLC_LC_prob + (1-conf.ignore_adj_veh_prob) * DLC_LC_prob
            else:
                E_LC_prob = DLC_LC_prob
            return E_LC_prob, "DLC", LC_related

    @staticmethod
    # @profile
    def Lateral_NDD(obs):
        """Decide the Lateral movement based on the NDD.

        Args:
            obs (dict): Observation of the surrounding vehicles.

        Returns:
            float: Probability of lane change.
            int: Type of lane change.
            tuple: Related information of the lane change.
        """
        initial_pdf = np.array([0,1,0])  # Left, Still, Right
        if not list(conf.OL_pdf):
            raise ValueError("No One Lead pdf file!")

        lane_id, v = obs["Ego"]["lane_index"], obs["Ego"]["velocity"]
        f1, r1, f0, r0, f2, r2 = obs["Lead"], obs["Foll"], obs["LeftLead"], obs["LeftFoll"], obs["RightLead"], obs["RightFoll"]

        if not f1:  # No vehicle ahead
            return False, 1, initial_pdf
        else:  # Has vehcile ahead
            left_prob, still_prob, right_prob = 0, 0, 0
            LC_related_list = []
            LC_type_list = []

            # Check NDD LC probability on both sides. Used to determine whether use MOBIL
            for item in ["Left", "Right"]:
                if item == "Left":
                    surrounding = (f1, f0, r0)
                    left_prob, LC_type, LC_related = NDDController._LC_prob(surrounding, obs)
                    LC_related_list.append(LC_related)
                    LC_type_list.append(LC_type)
                else:
                    surrounding = (f1, f2, r2)
                    right_prob, LC_type, LC_related = NDDController._LC_prob(surrounding, obs)
                    LC_related_list.append(LC_related)
                    LC_type_list.append(LC_type)
            has_LC_data_on_at_least_one_side_flag = True
            if left_prob is None and right_prob is None:
                has_LC_data_on_at_least_one_side_flag = False
            # If there is data on at least one side, then other side LC prob=0 even if there is no LC data on this side
            if has_LC_data_on_at_least_one_side_flag:
                if left_prob is None: 
                    left_prob = 0
                    right_prob = 2*right_prob
                elif right_prob is None: 
                    right_prob = 0
                    left_prob = 2*left_prob
            # Check whether there is CF data in this situation, if not or has no LC data on both sides then use MOBIL
            has_CF_data_flag = NDDController.check_whether_has_CF_data(obs["Ego"], f1)
            MOBIL_flag = ((not has_CF_data_flag) and (np.floor(v+0.5) <= 21)) or (not has_LC_data_on_at_least_one_side_flag)
            # MOBIL_flag = (not has_CF_data_flag) or (not has_LC_data_on_at_least_one_side_flag)
            # MOBIL_flag = (not has_CF_data_flag) or (np.floor(v+0.5) <= 30)
            
            # MOBIL            
            if MOBIL_flag:
                left_prob, right_prob = NDDController.MOBIL_result(obs)
                LC_related_list = [(v), (v)]
                LC_type_list = ["MOBIL", "MOBIL"]
                
            # In the leftest or rightest, double the other lane change probability
            if not obs["Ego"]["could_drive_adjacent_lane_left"]:
                left_prob = 0
                if conf.double_LC_prob_in_leftmost_rightest_flag:
                    right_prob *= 2
            if not obs["Ego"]["could_drive_adjacent_lane_right"]:
                right_prob = 0
                if conf.double_LC_prob_in_leftmost_rightest_flag:
                    left_prob *= 2
            if left_prob + right_prob > 1:
                tmp = left_prob + right_prob        
                left_prob *= 0.9/(tmp)
                right_prob *= 0.9/(tmp)                
            still_prob = 1 - left_prob - right_prob
            pdf_array = np.array([left_prob, still_prob, right_prob])
            if conf.safety_guard_enabled_flag:          
                pdf_array = Controller._check_lateral_safety(obs, pdf_array)                  
            
            lane_change_idx = np.random.choice(NDDController.LANE_CHANGE_INDEX_LIST,None,False,pdf_array)    
            to_lane_id = lane_id + lane_change_idx - 1
            if lane_change_idx != 1:
                return True, lane_change_idx, pdf_array
            else:
                return False, lane_change_idx, pdf_array

    @staticmethod 
    # @profile
    def round_to_(val, round_item, round_to_closest):
        """Round the val to the round_to_closest (for example 1.0, 0.2 ...)

        Args:
            val (float): Value to round.
            round_item (str): Item to round.
            round_to_closest (float): Round to the closest.

        Returns:
            float: Rounded value.
            int: Index of the rounded value.
        """        
        if round_item == "speed":
            value_list = conf.speed_list
        elif round_item == "range":
            value_list = conf.r_list
        elif round_item == "range_rate":
            value_list = conf.rr_list

        if round_to_closest == 1:
            mul, add, check = 1, 1, 0.5
        elif round_to_closest == 0.5:
            mul, add, check = 2, 0.5, 0.25        
        elif round_to_closest == 0.2:
            mul, add, check = 5, 0.2, 0.1

        if val < value_list[0]:
            val = value_list[0]
        elif val > value_list[-1]:
            val = value_list[-1]

        if conf.round_rule == "Round_to_closest":
            round_val = np.floor(val*mul+0.5)/mul
            try:
                assert(-check < round_val - val <= check+1e-10)
            except:
                round_val += add
                try:
                    assert(-check < round_val - val <= check+1e-10)
                except:
                    print(val, round_val)
                    raise ValueError("Round error!")
        
        try:
            round_idx = value_list.index(round_val)
        except:
            round_idx = min(range(len(value_list)), key=lambda i: abs(value_list[i]-round_val))
            assert(np.abs(value_list[round_idx]-round_val) < 1e-8)        

        return round_val, round_idx  

    @staticmethod 
    # @profile
    def _MOBIL_model(lc_decision, obs):
        """Mobil model for the NDD vehicle Lane change.

        Args:
            lc_decision: The candidate lane for the change.
            obs: Observation of surrounding vehicles.

        Returns:
            bool: Whether to do lane change.
            float: Gain for this lane change maneuver.
        """
        gain = None
        
        # Is the maneuver unsafe for the new following vehicle?
        if lc_decision == "left":
            new_preceding, new_following = obs["LeftLead"], obs["LeftFoll"]
        elif lc_decision == "right":
            new_preceding, new_following = obs["RightLead"], obs["RightFoll"]
        
        # Check whether will crash immediately
        r_new_preceding, r_new_following = 99999, 99999
        if new_preceding:
            r_new_preceding = new_preceding["distance"]
        if new_following:
            r_new_following = new_following["distance"]    
        if r_new_preceding <= 0 or r_new_following <= 0:
            return False, gain

        new_following_a = utils.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = utils.acceleration(ego_vehicle=new_following, front_vehicle=obs["Ego"])

        # The deceleration of the new following vehicle after the the LC should not be too big (negative)
        if new_following_pred_a < -conf.NDD_LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False, gain

        old_preceding, old_following = obs["Lead"], obs["Foll"]
        self_pred_a = utils.acceleration(ego_vehicle=obs["Ego"], front_vehicle=new_preceding)

        # Is there an acceleration advantage for me and/or my followers to change lane?
        self_a = utils.acceleration(ego_vehicle=obs["Ego"], front_vehicle=old_preceding)
        old_following_a = utils.acceleration(ego_vehicle=old_following, front_vehicle=obs["Ego"])
        old_following_pred_a = utils.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
        gain = self_pred_a - self_a + conf.NDD_POLITENESS * (new_following_pred_a - new_following_a + old_following_pred_a - old_following_a)
        if gain <= conf.NDD_LANE_CHANGE_MIN_ACC_GAIN:
            return False, gain
        return True, gain

    @staticmethod
    # @profile
    def MOBIL_result(obs):
        """Given the MOBIL model, calculate the left/right turn probability suggested by the MOBIL model.

        Args:
            obs: Observation of surrounding vehicles.

        Returns:
            float: Probability of left turn.
            float: Probability of right turn.
        """
        left_prob, right_prob = 0, 0
        if not conf.enable_MOBIL:
            return left_prob, right_prob        
        lane_id = obs["Ego"]["lane_index"]

        MOBIL_LC_prob = 0.99
        left_gain, right_gain = -np.inf, -np.inf
        left_LC_flag, right_LC_flag = False, False
        for lc_decision in ["left", "right"]:
            LC_flag, gain = NDDController._MOBIL_model(lc_decision, obs)
            if LC_flag:
                if lc_decision == "right": 
                    right_LC_flag, right_gain = LC_flag, gain
                elif lc_decision == "left": 
                    left_LC_flag, left_gain = LC_flag, gain
        
        # NEEDS Further discussion for the probability
        if left_LC_flag or right_LC_flag:
            if left_gain >= right_gain:
                left_prob, right_prob = MOBIL_LC_prob, 0.
            else:
                left_prob, right_prob = 0., MOBIL_LC_prob
            # assert(left_prob+right_prob == 1)
        return left_prob, right_prob

    @staticmethod
    # @profile
    def check_whether_has_CF_data(ego, f1):
        """If there is no CF data, then use IDM+MOBIL to decide the vehicle's maneuver.

        Args:
            ego: Information of the ego vehicle.
            f1: Information of the front vehicle.

        Returns:
            bool: Whether there is CF data.
        """
        v = ego["velocity"]
        r = f1["distance"]
        rr = f1["velocity"] - v
        round_speed, round_speed_idx = NDDController.round_to_(v, round_item="speed", round_to_closest=conf.v_resolution)
        round_r, round_r_idx = NDDController.round_to_(r, round_item="range", round_to_closest=conf.r_resolution)
        round_rr, round_rr_idx = NDDController.round_to_(rr, round_item="range_rate", round_to_closest=conf.rr_resolution)   

        pdf_array = conf.CF_pdf_array[round_r_idx, round_rr_idx, round_speed_idx]
        if sum(pdf_array) == 0:
            return False
        else:
            return True

    @staticmethod 
    # @profile
    def Longitudinal_NDD(obs, vehicle_length=conf.LENGTH):
        """Decide the Longitudinal acceleration based on the NDD.

        Args:
            obs (dict): Observation of the surrounding vehicles.
            vehicle_length (float): Length of the vehicle.
        
        Returns:
            float: Acceleration.
            np.array: Probability of different acceleration.
        """
        if not list(conf.CF_pdf_array):
            assert("No CF_pdf_array file!")
        if not list(conf.FF_pdf_array):
            assert("No FF_pdf_array file!")
        acc = 0
        ego_observation = obs["Ego"]
        # print(obs["Ego"]["velocity"])
        v = ego_observation["velocity"]
        f1 = obs["Lead"]

        if not f1:  # No vehicle ahead. Then FF
            round_speed, round_speed_idx = NDDController.round_to_(v, round_item="speed", round_to_closest=conf.v_resolution)
            pdf_array = conf.FF_pdf_array[round_speed_idx]                                                               
            if conf.safety_guard_enabled_flag:          
                pdf_array = Controller._check_longitudinal_safety(obs, pdf_array)                                                                                      
            return acc, pdf_array
        
        else:  # Has vehcile ahead. Then CF
            r = f1["distance"]
            rr = f1["velocity"] - v
            round_speed, round_speed_idx = NDDController.round_to_(v, round_item="speed", round_to_closest=conf.v_resolution)
            round_r, round_r_idx = NDDController.round_to_(r, round_item="range", round_to_closest=conf.r_resolution)
            round_rr, round_rr_idx = NDDController.round_to_(rr, round_item="range_rate", round_to_closest=conf.rr_resolution)            

            if not NDDController._check_bound_constraints(r, conf.r_low, conf.r_high) or not NDDController._check_bound_constraints(rr, conf.rr_low, conf.rr_high) or not NDDController._check_bound_constraints(v, conf.v_low, conf.v_high):
                pdf_array = NDDController.stochastic_IDM(ego_observation, f1)
                if conf.safety_guard_enabled_flag or conf.safety_guard_enabled_flag_IDM:
                    pdf_array = BaseController._check_longitudinal_safety(obs, pdf_array)            
                # acc = np.random.choice(conf.acc_list,None,False,pdf_array)
                return acc, pdf_array

            pdf_array = conf.CF_pdf_array[round_r_idx, round_rr_idx, round_speed_idx]
            if sum(pdf_array) == 0:
                # print("No CF data", round_speed, round_r, round_rr)
                pdf_array = NDDController.stochastic_IDM(ego_observation, f1)
                if conf.safety_guard_enabled_flag or conf.safety_guard_enabled_flag_IDM:
                    pdf_array = BaseController._check_longitudinal_safety(obs, pdf_array)              
                # acc = np.random.choice(conf.acc_list,None,False,pdf_array)       
                return acc, pdf_array
            if conf.safety_guard_enabled_flag:
                pdf_array = BaseController._check_longitudinal_safety(obs, pdf_array)              
            # acc = np.random.choice(conf.acc_list,None,False,pdf_array)        
            return acc, pdf_array        

    @staticmethod
    # @profile
    def _check_bound_constraints(value, bound_low, bound_high):
        """Check whether the value is within the bound.

        Args:
            value (float): Value to check.
            bound_low (float): Lower bound.
            bound_high (float): Upper bound.

        Returns:
            bool: Whether the value is within the bound.
        """
        if value < bound_low or value > bound_high:
            return False
        else:
            return True

    @staticmethod
    # @profile
    def stochastic_IDM(ego_vehicle, front_vehicle):
        """Stochastic IDM for the acceleration.

        Args:
            ego_vehicle (dict): Information of the ego vehicle.
            front_vehicle (dict): Information of the front vehicle.

        Returns:
            np.array: Probability of different acceleration.
        """
        tmp_acc = utils.acceleration(ego_vehicle=ego_vehicle, front_vehicle=front_vehicle)
        tmp_acc = np.clip(tmp_acc, conf.acc_low, conf.acc_high)
        acc_possi_list = scipy.stats.norm.pdf(conf.acc_list, tmp_acc, 0.3)
        # Delete possi if smaller than certain threshold
        acc_possi_list = [val if val > conf.Stochastic_IDM_threshold else 0 for val in acc_possi_list]
        assert(sum(acc_possi_list)>0)
        acc_possi_list = acc_possi_list/(sum(acc_possi_list))
        
        return acc_possi_list