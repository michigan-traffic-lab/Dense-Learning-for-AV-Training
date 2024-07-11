import sys
from .defaultconf import *
from .conf_rlagent import *

# av parameters
cav_agent = None
d2rl_flag = False
d2rl_criticality_threshold = 0.0
d2rl_slightlycritical = False
d2rl_subcritical_agent = None
load_cav_agent_flag = True
road_check_flag = True
RSS_flag = True
speedlimit_check_flag = True

# env parameters
criticality_threshold = 0 # 1e-4
env_mode = "NADE"
epsilon_value = 0.99 # 0.99, 1-1e-9
nade_criticality_threshold = 1e-4
load_nade_agent_flag = False
lane_change_amplify = 1.0 # 1.0, 0.01
max_lane_change_probability = 0.99
nade_agent = None
nade_offline_collection = False
precise_criticality_flag = True
precise_criticality_threshold = 0.9 # 0.01, 0.9, 0.07, 0.5
precise_weight_flag = False
precise_weight_threshold = 1.0
Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED = 8  # [m/s2]
traffic_flow_config["BV"] = True
train_mode = ""
treesearch_config["search_depth"] = 1
treesearch_config["surrogate_model"] = "surrogate"  # "AVI" "surrogate"
treesearch_config["offline_leaf_evaluation"] = False
treesearch_config["offline_discount_factor"] = 1
treesearch_config["treesearch_discount_factor"] = 1
weight_threshold = 0

# log parameters
compress_log_flag = True
computational_analysis_flag = False
debug_critical = False
log_mode = "offlinecollect"
more_info_critical = False

# experiment&simluation parameters
import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument("--yaml_path", type=str, default="yaml_configs/testing.yaml")
parser.add_argument("--worker_index", type=int, default=0)
args = parser.parse_args()
yaml_config = None
with open(args.yaml_path) as fp:
    yaml_config = yaml.load(fp, Loader=yaml.FullLoader)
assert yaml_config is not None
experiment_config = yaml_config["experiment_config"]
simulation_config = yaml_config["simulation_config"]
worker_index = args.worker_index
