import json, ujson
from .defaultconf import *
from .conf_rlagent import *

# av parameters
cav_agent = None
d2rl_flag = True
d2rl_criticality_threshold = 0.0
d2rl_slightlycritical = False
d2rl_subcritical_agent = None
load_cav_agent_flag = True
road_check_flag = True
RSS_check_flag = False
RSS_flag = True
speedlimit_check_flag = True

# env parameters
criticality_threshold = 0 # 1e-4
env_mode = "NADE"
epsilon_value = 1-1e-9 # 0.99
load_nade_agent_flag = False
lane_change_amplify = 1.0 # 1.0, 0.01
max_lane_change_probability = 0.99
nade_agent = None
precise_criticality_flag = True
precise_criticality_threshold = 0.9 # 0.01 , 0.9
precise_weight_threshold = 0.0
Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED = 8  # [m/s2]
traffic_flow_config["BV"] = True
train_mode = "offline"
train_case_study_mode = "crash_nearmiss"
treesearch_config["search_depth"] = 1
treesearch_config["surrogate_model"] = "surrogate"  # "AVI" "surrogate"
treesearch_config["offline_leaf_evaluation"] = False
treesearch_config["offline_discount_factor"] = 1
treesearch_config["treesearch_discount_factor"] = 1
update_crash = True
weight_threshold = 0

# log parameters
computational_analysis_flag = False
debug_critical = False

# experiment&simluation parameters
import yaml
if "avtraining_yaml_path" in os.environ:
    yaml_path = os.environ["avtraining_yaml_path"]
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, default="yaml_configs/training.yaml")
    args = parser.parse_args()
    yaml_path = args.yaml_path
yaml_config = None
with open(yaml_path) as fp:
    yaml_config = yaml.load(fp, Loader=yaml.FullLoader)
assert yaml_config is not None
experiment_config = yaml_config["experiment_config"]
simulation_config = yaml_config["simulation_config"]
simulation_config["neuralmetric_config"]["ckpt"] = os.path.join(experiment_config["code_root_folder"], simulation_config["neuralmetric_config"]["ckpt"])
simulation_config["pytorch_model_path_list"][0] = os.path.join(experiment_config["code_root_folder"], simulation_config["pytorch_model_path_list"][0])
simulation_config["pytorch_nade_model_path"] = os.path.join(experiment_config["code_root_folder"], simulation_config["pytorch_nade_model_path"])

data_folder = experiment_config["data_folder"]
data_info_origin_path = os.path.join(data_folder[0], "offline_av_alldata_new.json")
data_info_new_path = os.path.join(data_folder[0], "offline_av_neweval_crashnearmiss_new.json")
data_info_origin = None
data_folder_additional_info = None
data_info_weight1_ratio = 0.5 # 5.139275315774029e-07/3.0313606351423044e-06
if data_info_origin is None:
    if isinstance(data_info_origin_path, list) and len(data_info_origin_path) > 1:
        data_info_origin = []
        for p_ in data_info_origin_path:
            with open(p_) as file_obj:
                data_info_origin_part = ujson.load(file_obj)
            data_info_origin_part["safe_ep_info"] = [tuple(info) for info in data_info_origin_part["safe_ep_info"]]
            data_info_origin.append(data_info_origin_part)
    else: # data_info_origin_path is a string
        with open(data_info_origin_path) as file_obj:
            data_info_origin = ujson.load(file_obj)
        data_info_origin["safe_ep_info"] = [tuple(info) for info in data_info_origin["safe_ep_info"]]
    # print("load data_info_origin")
eval_round = ep_num_eval_worker = experiment_config["ep_num_eval_worker"]
eval_three_circle_min_distance_threshold = experiment_config["eval_three_circle_min_distance_threshold"]
number_eval_workers = experiment_config["number_eval_workers"]
