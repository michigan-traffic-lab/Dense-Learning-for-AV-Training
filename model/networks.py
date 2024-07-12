import torch
import torch.nn as nn
# from bert import Embeddings, Block, Config
torch.manual_seed(0)

def normalize(a):
    """Normalize the input tensor.

    Args:
        a (torch.Tensor): The input tensor.
    
    Returns:
        torch.Tensor: The normalized tensor.
    """
    for i in range(len(a)):
        a[i][::4] = a[i][::4] - a[i][0]
        a[i][1::4] = a[i][1::4] - 46
        a[i][2::4] = a[i][2::4] - 30
        a[i][3::4] = a[i][3::4] - 90
        a[i] = torch.clip(a[i], -20, 20)/20.0
    return a

def define_G(yaml_conf, input_dim, output_dim, device):
    """Define the hidden layers for the neural network.

    Args:
        yaml_conf (dict): The configuration dictionary.
        input_dim (int): The input dimension.
        output_dim (int): The output dimension.
        device (torch.device): The device to use.

    Returns:
        nn.Module: The neural network model.
    """
    if "input_dim" in yaml_conf:
        input_dim = yaml_conf["input_dim"]
    if "output_dim" in yaml_conf:
        output_dim = yaml_conf["output_dim"]
        
    if yaml_conf["model"] == 'linear':
        return SimpleLinearModel(yaml_conf, input_dim, output_dim).to(device)
    elif yaml_conf["model"] == 'simple_mlp':
        return SimpleMLP(yaml_conf, input_dim, output_dim).to(device)
    elif yaml_conf["model"] == 'bn_mlp':
        return BnMLP(yaml_conf, input_dim, output_dim).to(device)
    elif yaml_conf["model"] == "bn_mlp_sigmoid":
        return BnMLP_sigmoid(yaml_conf, input_dim, output_dim).to(device)
    elif yaml_conf["model"] == "bnmlp_sigmoid_simplified":
        return  BnMLP_sigmoid_simplified(yaml_conf, input_dim, output_dim).to(device)
    elif yaml_conf["model"] == 'transformer':
        bert_cfg = Config()
        bert_cfg.dim = 256
        bert_cfg.max_len = 9
        return Transformer(input_d_token=3, output_d_token=3, cfg=bert_cfg).to(device)
    else:
        raise NotImplementedError(
            'Wrong model name %s (choose one from [lsr, softmax])' % yaml_conf["model"])

class SimpleLinearModel(nn.Module):

    def __init__(self, yaml_conf, input_dim, output_dim):
        """Initialize the SimpleLinearModel.

        Args:
            yaml_conf (dict): The configuration information.
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.
        """
        super().__init__()
        self.yaml_conf = yaml_conf
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)

    def forward(self, x):
        """Forward function of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # x: batch_size x feat_dim
        if "normalize" in self.yaml_conf and self.yaml_conf["normalize"]:
            x = normalize(x)
        logits = self.linear(x)
        return logits


class SimpleMLP(nn.Module):

    def __init__(self, yaml_conf, input_dim, output_dim):
        """Initialize the SimpleMLP.

        Args:
            yaml_conf (dict): The configuration information.
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.
        """
        super().__init__()
        self.yaml_conf = yaml_conf
        self.linear1 = nn.Linear(in_features=input_dim, out_features=256, bias=True)
        self.linear2 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.linear3 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.linear4 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.linear_pred = nn.Linear(in_features=64, out_features=output_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward function of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # x: batch_size x feat_dim
        if "normalize" in self.yaml_conf and self.yaml_conf["normalize"]:
            x = normalize(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        logits = self.linear_pred(x)

        return logits



class BnMLP(nn.Module):

    def __init__(self, yaml_conf, input_dim, output_dim):
        """Initialize the BnMLP.

        Args:
            yaml_conf (dict): The configuration information.
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.
        """
        super().__init__()
        self.yaml_conf = yaml_conf
        self.linear1 = nn.Linear(in_features=input_dim*4, out_features=256, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.bn4 = nn.BatchNorm1d(num_features=64)
        self.linear_pred = nn.Linear(in_features=64, out_features=output_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward function of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # x: batch_size x feat_dim
        if "normalize" in self.yaml_conf and self.yaml_conf["normalize"]:
            x = normalize(x)
        x = torch.cat([x, x**2, torch.sin(x), torch.cos(x)], dim=-1)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.relu(self.bn4(self.linear4(x)))
        logits = self.linear_pred(x)

        return logits


class BnMLP_sigmoid(nn.Module):

    def __init__(self, yaml_conf, input_dim, output_dim):
        """Initialize the BnMLP_sigmoid.

        Args:
            yaml_conf (dict): The configuration information.
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.
        """
        super().__init__()
        self.yaml_conf = yaml_conf
        self.linear1 = nn.Linear(in_features=input_dim*4, out_features=256, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.bn4 = nn.BatchNorm1d(num_features=64)
        self.linear_pred = nn.Linear(in_features=64, out_features=output_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward function of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # x: batch_size x feat_dim
        if "normalize" in self.yaml_conf and self.yaml_conf["normalize"]:
            x = normalize(x)
        x = torch.cat([x, x**2, torch.sin(x), torch.cos(x)], dim=-1)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.relu(self.bn4(self.linear4(x)))
        logits = self.linear_pred(x)
        logits = self.sigmoid(logits)

        return logits

    def get_last_shared_layer(self):
        """Get the last shared layer.

        Returns:
            nn.Module: The last shared layer.
        """
        return self.linear_pred

class BnMLP_sigmoid_simplified(nn.Module):

    def __init__(self, yaml_conf, input_dim, output_dim):
        """Initialize the BnMLP_sigmoid_simplified.

        Args:
            yaml_conf (dict): The configuration information.
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.
        """
        super().__init__()
        self.yaml_conf = yaml_conf
        self.linear1 = nn.Linear(in_features=input_dim*4, out_features=64, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.linear_pred = nn.Linear(in_features=64, out_features=output_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward function of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # x: batch_size x feat_dim
        if "normalize" in self.yaml_conf and self.yaml_conf["normalize"]:
            x = normalize(x)
        x = torch.cat([x, x**2, torch.sin(x), torch.cos(x)], dim=-1)
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        logits = self.linear_pred(x)
        logits = self.sigmoid(logits)
        return logits
    
    def get_last_shared_layer(self):
        """Get the last shared layer.

        Returns:
            nn.Module: The last shared layer.
        """
        return self.linear_pred

class BnMLP_sigmoid_single_layer(nn.Module):

    def __init__(self, yaml_conf, input_dim, output_dim):
        """Initialize the BnMLP_sigmoid_single_layer.

        Args:
            yaml_conf (dict): The configuration dictionary.
            input_dim (int): The input dimension.
            output_dim (int): The output dimension.
        """
        super().__init__()
        self.yaml_conf = yaml_conf
        self.linear1 = nn.Linear(in_features=input_dim*4, out_features=64, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.linear_pred = nn.Linear(in_features=64, out_features=output_dim, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward function of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # x: batch_size x feat_dim
        if "normalize" in self.yaml_conf and self.yaml_conf["normalize"]:
            x = normalize(x)
        x = torch.cat([x, x**2, torch.sin(x), torch.cos(x)], dim=-1)
        x = self.relu(self.bn1(self.linear1(x)))
        logits = self.linear_pred(x)
        logits = self.sigmoid(logits)
        return logits
    
    def get_last_shared_layer(self):
        """Get the last shared layer.

        Returns:
            nn.Module: The last shared layer.
        """
        return self.linear_pred
        

class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    """

    def __init__(self, input_dim, L=5, pos_scale=0.01, heading_scale=1.0):
        """Initialize the PositionalMapping.

        Args:
            input_dim (int): The input dimension.
            L (int, optional): The number of layers. Defaults to 5.
            pos_scale (float, optional): The scale for the position. Defaults to 0.01.
            heading_scale (float, optional): The scale for the heading. Defaults to 1.0.
        """
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        # self.scale = scale
        self.pos_scale = pos_scale
        self.heading_scale = heading_scale

    def forward(self, x):
        """Forward function of the model.

        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output tensor.
        """
        if self.L == 0:
            return x

        # x = x * self.scale
        x_scale = x.clone()
        x_scale[:, :, :10] = x_scale[:, :, :10] * self.pos_scale  # Fixme: hardcode first 10 dimensions are position, and last half are cos and sin of heading.
        x_scale[:, :, 10:] = x_scale[:, :, 10:] * self.heading_scale

        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x_scale)
            x_cos = torch.cos(2**i * PI * x_scale)

            x_sin[:, :, :10], x_sin[:, :, 10:] = x_sin[:, :, :10] / self.pos_scale, x_sin[:, :, 10:] / self.heading_scale
            x_cos[:, :, :10], x_cos[:, :, 10:] = x_cos[:, :, :10] / self.pos_scale, x_cos[:, :, 10:] / self.heading_scale

            h.append(x_sin)
            h.append(x_cos)

        # return torch.cat(h, dim=-1) / self.scale
        return torch.cat(h, dim=-1)

class PredictionsHeads(nn.Module):
    """
    Prediction layer with two output heads, one modeling mean and another one modeling std.
    Also prediction cos and sin headings.
    """

    def __init__(self, h_dim, output_dim):
        """Initialize the PredictionsHeads.

        Args:
            h_dim (int): The hidden dimension.
            output_dim (int): The output dimension.
        """
        super().__init__()
        # x, y position
        self.out_net_mean = nn.Linear(in_features=h_dim, out_features=int(output_dim/2), bias=True)
        self.out_net_std = nn.Linear(in_features=h_dim, out_features=int(output_dim/2), bias=True)
        self.elu = torch.nn.ELU()

        # cos and sin heading
        self.out_net_cos_sin_heading = nn.Linear(in_features=h_dim, out_features=int(output_dim/2), bias=True)

    def forward(self, x):
        """Forward function of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple: The output mean, std, and cos and sin heading.
        """
        # shape x: batch_size x m_token x m_state
        out_mean = self.out_net_mean(x)
        out_std = self.elu(self.out_net_std(x)) + 1

        out_cos_sin_heading = self.out_net_cos_sin_heading(x)

        return out_mean, out_std, out_cos_sin_heading