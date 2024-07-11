# Data Collection/Evaluation

## 1. Collect training data/Evaluate
Please run the following command to collect data and evaluate the agent's safety performance:
```bash
python main_testing.py
```
The file structure of the evaluation results is shown below:
```
Dense-Learning-for-AV-Training/output/Experiment-testing_DateOfRunning/
|__crash/
|__leave_network/
|__rejected/
|__speed_out_of_range/
|__tested_and_safe/
|__weight0.npy
```
Since the evaluation process is time-consuming, we highly recommend running the evaluation process using parallel processing:
```bash
bash main_testing_mp.sh -n 4
# where `-n` specifies the number of parallel processes.
```

## 2. (Optional) Modify the configuration file
Suppose you want to change the experiment name, modify the path to store the evaluation results, or evaluate a different AV algorithm, please go to the [configuration file](yaml_configs/testing.yaml) and modify the following variables accordingly:

a. Modify the experiment name by changing the **experiment_name**.
```yaml
...
experiment_config:
  ...
  experiment_name: "your_experiment_name"
  ...
```

b. Modify the path to store the evaluation results by changing the **root_folder**.
```yaml
...
experiment_config:
  ...
  root_folder: "your_path_to_store_evaluation_results"
  ...
```

c. If you have trained a SafeDriver and want to evaluate its safety performance, please modify the path to the SafeDriver by changing the second element of the **pytorch_model_path_list**.
```yaml
simulation_config:
  ...
  pytorch_model_path_list: 
    - "./agents/AV/basemodel_ckpt369.pt"
    - "path_to_your_trained_model"
  ...
```

<- Last Page: [Installation](installation.md)

-> Next Page: [Preparing Dataset](prepare_dataset.md)