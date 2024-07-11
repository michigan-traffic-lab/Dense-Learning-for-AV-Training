import os
import numpy as np
import scipy.stats
from .nddcontroller import NDDController
import utils
if os.environ["mode"] == "testing":
    import conf.conf_testing as conf
elif os.environ["mode"] == "training":
    import conf.conf_training as conf
else:
    raise ValueError("Please set the mode to testing or training")

class NADEBackgroundController(NDDController):
    def __init__(self):
        """Initialize the NADEBackgroundController class.
        """        
        super().__init__(controllertype="NADEBackgroundController")
        self.weight = None
        self.ndd_possi = None
        self.critical_possi = None
        self.epsilon_pdf_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.normalized_critical_pdf_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.ndd_possi_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.bv_criticality_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.bv_challenge_array = np.zeros(len(conf.ACTIONS), dtype=float)

    def _type(self):
        """Obtain the type of the controller.

        Returns:
            str: Type of the controller.
        """        
        return 'NADEBackgroundController'

    # @profile
    def get_NDD_possi(self):
        """Obtain the possibility of the BV maneuvers.

        Returns:
            np.array: Possibility of the BV maneuvers.
        """
        return np.array(self.ndd_pdf)

    # @profile
    def _sample_critical_action(self, bv_criticality, criticality_array, ndd_possi_array, epsilon=conf.epsilon_value):
        """Sample critical action of the controlled BV after doing the epsilon-greedy.

        Args:
            criticality_array (list(float)): List of criticality of each BV maneuver.
            bv_criticality (float): Total criticality of the studied BV.
            possi_array (list(float)): List of probability of each BV maneuver.

        Returns:
            integer: Sampled BV action index.
            float: Weight of the BV action.
            float: Possibility of the BV action.
            float: Criticality of the BV action.
        """        
        normalized_critical_pdf_array = criticality_array / bv_criticality
        epsilon_pdf_array = utils.epsilon_greedy(normalized_critical_pdf_array, ndd_possi_array, epsilon=epsilon)
        bv_action_idx = np.random.choice(len(conf.BV_ACTIONS), 1, replace=False, p=epsilon_pdf_array)
        critical_possi, ndd_possi = epsilon_pdf_array[bv_action_idx], ndd_possi_array[bv_action_idx]
        weight_list = (ndd_possi_array+1e-30)/(epsilon_pdf_array+1e-30)
        weight = ndd_possi/critical_possi
        self.bv_criticality_array = criticality_array
        self.normalized_critical_pdf_array = normalized_critical_pdf_array
        self.ndd_possi_array = ndd_possi_array
        self.epsilon_pdf_array = epsilon_pdf_array
        self.weight = weight.item()
        self.ndd_possi = ndd_possi.item()
        self.critical_possi = critical_possi.item()
        return bv_action_idx, weight, ndd_possi, critical_possi, weight_list

    @staticmethod
    # @profile
    def _hard_brake_challenge(v, r, rr):     
        """Calculate the hard brake challenge value. 
           Situation: BV in front of the CAV, do the hard-braking.

        Args:
            v (float): Speed of BV.
            r (float): Distance between BV and CAV.
            rr (float): Range rate of BV and CAV.

        Returns:
            list(float): List of challenge for the BV behavior.
        """
        CF_challenge_array = np.zeros((len(conf.BV_ACTIONS)-2), dtype=float)
        round_speed, round_r, round_rr = utils._round_data_plain(v, r, rr) 
        index = np.where((conf.CF_state_value == [round_r, round_rr, round_speed]).all(1))
        assert(len(index) <= 1)
        index = index[0]
        if len(index):
            CF_challenge_array = conf.CF_challenge_value[index.item(), :]
        return CF_challenge_array
    
    @staticmethod
    # @profile
    def _BV_accelerate_challenge(v, r, rr):
        """Assume the CAV is cutting in the BV and calculate by the BV CF

        Args:
            v (float): Speed of BV.
            r (float): Distance between BV and CAV.
            rr (float): Range rate between BV and CAV.

        Returns:
            float: Challenge of the BV behavior.
        """
        BV_CF_challenge_array = np.zeros((len(conf.BV_ACTIONS) - 2), dtype=float)
        new_r = r + rr
        if new_r <= 4.1:
            BV_CF_challenge_array = np.ones(
            (len(conf.BV_ACTIONS) - 2), dtype=float)

        return BV_CF_challenge_array

    def Decompose_decision(self, CAV, SM_LC_prob, full_obs=None, predicted_full_obs=None, predicted_traj_obs=None):
        """Decompose decision of the studied BV.

        Args:
            CAV (dict): Observation of the CAV.
            SM_LC_prob (list(float)): List of possibility for the CAV from the surrogate model.

        Returns:
            float: Total criticality of the BV.
            integer: BV action index.
            float: Weight of the BV action.
            float: Possibility of the BV action.
            float: Criticality of the BV action.
        """
        self.bv_criticality_array = np.zeros(len(conf.ACTIONS), dtype=float)
        bv_criticality, bv_action_idx, weight, ndd_possi, critical_possi, weight_list, criticality_array = - \
            np.inf, None, None, None, None, None, np.zeros(len(conf.ACTIONS), dtype=float)
        bv_id = self.vehicle.id
        bv_pdf = self.get_NDD_possi()
        bv_obs = self.vehicle.observation.information
        bv_left_prob, bv_right_prob = bv_pdf[0], bv_pdf[1]
        if 1:
            bv_criticality, criticality_array, bv_challenge_array, risk = self._calculate_criticality(
                bv_obs, CAV, SM_LC_prob, full_obs, predicted_full_obs, predicted_traj_obs)
            self.bv_challenge_array = bv_challenge_array
        return bv_criticality, criticality_array
    
    def Decompose_sample_action(self, bv_criticality, bv_criticality_array, bv_pdf, epsilon=conf.epsilon_value):
        """Decompose the sampled action of the BV.

        Args:
            bv_criticality (float): Total criticality of the BV.
            bv_criticality_array (list(float)): List of criticality of each BV maneuver.
            bv_pdf (list(float)): List of probability of each BV maneuver.
            epsilon (float, optional): Epsilon value for the epsilon-greedy. Defaults to conf.epsilon_value.

        Returns:
            integer: BV action index.
            float: Weight of the BV action.
            float: Possibility of the BV action.
            float: Criticality of the BV action.
        """
        if epsilon is None:
            epsilon = conf.epsilon_value
        bv_action_idx, weight, ndd_possi, critical_possi, weight_list = None, None, None, None, None
        if bv_criticality > conf.criticality_threshold:
            bv_action_idx, weight, ndd_possi, critical_possi, weight_list = self._sample_critical_action(
                bv_criticality, bv_criticality_array, bv_pdf, epsilon)
        if weight is not None:
            weight, ndd_possi, critical_possi = weight.item(
            ), ndd_possi.item(), critical_possi.item()
        return bv_action_idx, weight, ndd_possi, critical_possi, weight_list