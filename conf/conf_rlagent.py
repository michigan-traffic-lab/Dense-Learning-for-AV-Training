import numpy as np
import torch


def load_ray_agent(checkpoint_path_list, slightly_critical_ckpt=[]):
    """Loads the agent from a list of checkpoint paths.

    Args:
        checkpoint_path_list (list): A list of strings, where each string is a path to a model checkpoint.
        slightly_critical_ckpt (list, optional): A list of model checkpoints for the slightly safety-critical states. Defaults to an empty list.

    Returns:
        MixedAgent: An instance of the MixedAgent class initialized with the models loaded from the provided checkpoint paths.
    """
    discriminator_agent = MixedAgent(checkpoint_path_list, slightly_critical_ckpt)
    return discriminator_agent


class MixedAgent:
    def __init__(self, checkpoint_path_list, slightly_critical_ckpt=[]):
        """Initializes the MixedAgent with models loaded from the specified checkpoint paths.

        Args:
            checkpoint_path_list (list): A list of strings, where each string is a path to a model checkpoint.
            slightly_critical_ckpt (list, optional): A list of model checkpoints for the slightly safety-critical states. Defaults to an empty list.
        """
        self.agent_list = []
        for path in checkpoint_path_list:
            print("load pytorch model from ",path)
            model = torch.jit.load(path)
            model.eval()
            self.agent_list.append(model)

    # @profile
    def compute_action(self, observation, highly_critical=False, slightly_critical=False):
        """Computes the action to be taken based on the observation and criticality of the situation.

        Args:
            observation (array): The current observation that the agent needs to act upon.
            highly_critical (bool, optional): A flag indicating if the situation is highly critical. Defaults to False.
            slightly_critical (bool, optional): A flag indicating if the situation is slightly critical. Defaults to False.

        Returns:
            np.array: The action computed by the selected agent based on the observation and criticality flags.
        """
        if len(self.agent_list) == 1:
            agent = self.agent_list[0]
        elif len(self.agent_list) == 2:
            # if highly_critical or slightly_critical:
            if highly_critical:
                agent = self.agent_list[1]
            else:
                agent = self.agent_list[0]
        elif len(self.agent_list) == 3:
            if slightly_critical:
                # print("find slightly")
                agent = self.agent_list[2]
            elif highly_critical:
                # print("find highly")
                agent = self.agent_list[1]
            else:
                # print("find normal")
                agent = self.agent_list[0]
        else:
            raise ValueError("Too many/few RL agents: number=",
                             len(self.agent_list))
        # action_result = agent.compute_single_action(observation)
        o = torch.reshape(torch.tensor(observation), (1,len(observation)))
        out = agent({"obs":o},[torch.tensor([0.0])],torch.tensor([1]))
        action = out[0][0]
        action_result = np.array([np.clip((float(action[0])+1)*3-4,-4.,2.), np.clip((float(action[1])+1)*10-10,-10.,10.)])
        # print(action_result_list, statistics.mean(action_result_list))
        return action_result


class torch_discriminator_agent:
    def __init__(self, load_path):
        """Initializes the torch_discriminator_agent with a model loaded from the specified path.

        This constructor loads a PyTorch model from a given file path and sets it to evaluation mode.

        Args:
            load_path (str): The file path from which to load the PyTorch model.
        """
        print("load nade model from", load_path)
        self.model = torch.jit.load(load_path)
        self.model.eval()

    def compute_action(self, observation):
        """Computes the action for the given observation using the discriminator model.

        This method processes the observation, feeds it to the discriminator model, and processes the output to
        produce a final action. The action is adjusted to ensure it falls within specified bounds.

        Args:
            observation (list): The current state observation, a list of floats representing the environment state.

        Returns:
            float: The computed action, clipped to ensure it falls within a predefined range.
        """
        lb = 0.00001
        ub = 0.99999
        obs = torch.reshape(torch.tensor(observation), (1,len(observation)))
        out = self.model({"obs":obs},[torch.tensor([0.0])],torch.tensor([1]))
        action = np.clip((float(out[0][0][0])+1)*(ub-lb)/2 + lb, lb, ub)
        return action


def load_discriminator_agent(pytorch_nade_model_path): #mode="ray"
    """Loads a discriminator agent from a specified PyTorch model path.

    This function initializes a torch_discriminator_agent with the model loaded from the given path.

    Args:
        pytorch_nade_model_path (str): The file path of the PyTorch model to load.

    Returns:
        torch_discriminator_agent: An instance of torch_discriminator_agent initialized with the loaded model.
    """
    discriminator_agent = torch_discriminator_agent(pytorch_nade_model_path)
    return discriminator_agent