import numpy as np
import os
import torch
import yaml

from model.networks import *


class NN_Metric():
    def __init__(self, yaml_conf):
        """Initialize the NN_Metric class.

        Args:
            yaml_conf (dict): Configuration information from the yaml file.

        Raises:
            NotImplementedError: If the pre-trained weights does not exist.
        """
        self.device = torch.device("cpu")
        self.yaml_conf = yaml_conf
        
        # Load checkpoint
        checkpoint = self.yaml_conf["ckpt"]
        # print(checkpoint)
        self.net_G = define_G(yaml_conf = self.yaml_conf, input_dim=28, output_dim=1, device=self.device)
        if os.path.exists(checkpoint):
            checkpoint = torch.load(checkpoint, map_location=self.device)
            # print(f"Best checkpoint epoch id {checkpoint['best_epoch_id']} with accuracy {checkpoint['best_val_acc']}")
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.eval()
        else:
            raise NotImplementedError(
                'pre-trained weights %s does not exist...' % os.path.join(self.yaml_conf["ckpt"], 'best_ckpt.pt'))

    def inference(self, x):
        """Inference the neural network model.

        Args:
            x (np.array): Input of the neural network model.

        Returns:
            float: Output of the neural network model, which is the criticality of the current situation.
        """
        x = torch.Tensor(x).to(self.device)
        # Apply the inference
        y_pred = self.net_G(x)
        return float(y_pred.detach().cpu().squeeze().numpy())

    def normalize(self, obs):
        """Normalize the observation.

        Args:
            obs (np.array): Observation of the environment.

        Returns:
            np.array: Normalized observation.
        """
        if self.yaml_conf["obs_normalize"]:
            # center at CAV
            noneveh_index = np.where(obs == -1)
            obs[:, 0::4][:, 1:] = obs[:, 0::4][:, 1:] - np.repeat(obs[:, 0].reshape(-1, 1), 6, axis=1)
            obs[:, 1::4][:, 1:] = obs[:, 1::4][:, 1:] - np.repeat(obs[:, 1].reshape(-1, 1), 6, axis=1)
            obs[noneveh_index] = -1

            # normalize the obs from -1 to 1
            cav_lb = [400, 42, 20, 80]
            cav_ub = [900, 50, 40, 100]
            relative_bv_lb = [-30, -8, 20, 80]
            relative_bv_ub = [30, 8, 40, 100]
            total_lb = cav_lb + relative_bv_lb * 6
            total_ub = cav_ub + relative_bv_ub * 6
            total_lb = np.array(total_lb)
            total_ub = np.array(total_ub)

            # obs = (obs - total_lb) / (total_ub - total_lb) * 2 - 1
            # # constrain the obs to be in the range of -1 to 1
            # obs = np.clip(obs, -1, 1)
            # new normalize, out of bound vehicles are [-1]*4, others follow default normalize

            _1_bv_outbound = np.where(abs(obs[:,4])>relative_bv_ub[0])
            _2_bv_outbound = np.where(abs(obs[:,8])>relative_bv_ub[0])
            _3_bv_outbound = np.where(abs(obs[:,12])>relative_bv_ub[0])
            _4_bv_outbound = np.where(abs(obs[:,16])>relative_bv_ub[0])
            _5_bv_outbound = np.where(abs(obs[:,20])>relative_bv_ub[0])
            _6_bv_outbound = np.where(abs(obs[:,24])>relative_bv_ub[0])

            obs = (obs - total_lb) / (total_ub - total_lb) * 2 - 1
            # constrain the obs to be in the range of -1 to 1
            obs = np.clip(obs, -1, 1)
            obs[_1_bv_outbound,4:8] = np.array([-1]*4).reshape(1,-1)
            obs[_2_bv_outbound,8:12] = np.array([-1]*4).reshape(1,-1)
            obs[_3_bv_outbound,12:16] = np.array([-1]*4).reshape(1,-1)
            obs[_4_bv_outbound,16:20] = np.array([-1]*4).reshape(1,-1)
            obs[_5_bv_outbound,20:24] = np.array([-1]*4).reshape(1,-1)
            obs[_6_bv_outbound,24:28] = np.array([-1]*4).reshape(1,-1)
            obs[noneveh_index] = -1

            obs = obs[:,1:]
        
        return obs