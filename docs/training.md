# Training SafeDriver

## 1. Modify the configuration
Before training the SafeDriver in the three-lane highway scenario, please update certain parameters inside the [default configuration file](yaml_configs/training.yaml). To be more specific, you will need to modify the following variables:
```yaml
...
experiment_config:
  ...
  data_folder: 
    - "/path/to/training/data/folder" # please set this variable to be the absolute path of the training data folder
  ...
  experiment_name: "training" # please set this variable to be any experiment name you want
  ...
  number_eval_workers: 200 # please set this variable to be the number of cpu for evaluating the latest agent after one iteration of training
  num_workers: 179 # please set this variable to be the number of cpu for collecting the training batches
  ...
  restore_path: "/path/to/source/code/agents/AV/checkpoint_000369/checkpoint-369" # please set this variable to be the path of the checkpoint you want to initialize from
  root_folder: "/path/to/store/training/results" # please set this variable to be the path where you want to save the training results
  ...
```

## 2. Train
Please run the following command within the root folder to train the SafeDriver and continuously improve its performance:
```bash
python main_training.py
```
The training results will be stored in the folder defined in the configuration file, which has the following structure:
```
/path/to/store/training/results/experiment_name/
|__PPO_my_env_*/
|_____checkpoint_*/
|_____events.out.*
|_____params.json
|_____params.pkl
|_____progress.csv
|_____result.json
|__basic-variant-state-*.json
|__experiment_state-*.json
```

## 3. Export the trained model
After training, you should export the trained model as a PyTorch model using the following code
 ```bash
# Please remember to update the three arguments: 
# "model_root_folder": This argument defines the path to the training results, it should look like "Example_Training_Iteration/training_results/test_training/PPO_my_env_c339c_00000_0_2024-01-18_06-43-14"; 
# "ckpt_num": This argument defines the checkpoint number you want to export, it should be the number of the checkpoint you want to export, e.g., 394;
# "det_folder": This argument defines the path to save the exported model, it should look like "Example_Training_Iteration/training_results".
python export_densemodel.py --model_root_folder /path/to/training/results --ckpt_num 394 --det_folder /path/to/save/model
```

> Based on our experience, going through the whole process, including data collection, data processing and training, will need more than 4000 core*hours in total.

In summary, to reproduce the quantitative results in the manuscript, it is advised that a minimum of five iterations be performed to train the SafeDriver. In each iteration, please first collect enough crashes (at least 500) and near-miss, aiming to form an effective training dataset. Following dataset construction, you can train the SafeDriver until the model's performance stabilizes, typically after about thirty cycles. Then, use the resultant model as a starting point for the subsequent training iteration, which again starts with data collection.

<- Last Page: [Preparing Dataset](prepare_dataset.md)

-> Next Page: [Demonstration](demonstration.md)