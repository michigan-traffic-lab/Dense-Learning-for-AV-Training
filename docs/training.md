# Training SafeDriver

## 1. Modify the configuration
Before training the SafeDriver in the three-lane highway scenario, please update certain parameters inside the [default configuration file](../yaml_configs/training.yaml). To be more specific, you will need to modify the following variables:
```yaml
...
experiment_config:
  ...
  code_root_folder: "/absolute/path/to/Dense-Learning-for-AV-Training" # please set this variable to be the absolute path of the source code folder
  data_folder: 
    - "/absolute/path/to/training/data/folder" # please set this variable to be the absolute path of the training data folder, which should look like this "/home/user/Example_Training_Iteration/testing_results/Experiment-test_basemodel_2024-01-17/densified_exps"
  ...
  ep_num_eval_worker: 100 # please set this variable to be the number of iterations for the evaluation process
  ...
  experiment_name: "training" # please set this variable to be any experiment name you want
  ...
  number_eval_workers: 200 # please set this variable to be the number of cpu for evaluating the latest agent after one iteration of training
  num_workers: 179 # please set this variable to be the number of cpu for collecting the training batches
  ...
  restore_path: "/absolute/path/to/restored/checkpoint" # please set this variable to be the absolute path of the checkpoint you want to initialize from, which should look like this "/home/user/Dense-Learning-for-AV-Training/agents/AV/checkpoint_000369/checkpoint-369"
  root_folder: "/absolute/path/to/store/training/results" # please set this variable to be the absolute path where you want to save the training results
  ...
```
> Please make sure the product of "**ep_num_eval_worker**" and "**number_eval_workers**" is larger than the total number of crashes and near-miss events in the training data folder, so that enough collected data are evaluated after each training step.

## 2. Train
Please run the following command within the root folder to train the SafeDriver and continuously improve its performance:
```bash
python main_training.py
```
The training results will be stored in the folder defined in the configuration file, which has the following structure:
```
/absolute/path/to/store/training/results/experiment_name/
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
python export_densemodel.py --model_root_folder /absolute/path/to/store/training/results/experiment_name/PPO_my_env_*/ --ckpt_num 394 --det_folder /path/to/save/model
```

> Based on our experience, going through the whole process, including data collection, data processing and training, will need more than 4000 core*hours in total.

In summary, to reproduce the quantitative results in the manuscript, it is advised that a minimum of five iterations be performed to train the SafeDriver. In each iteration, please first collect enough crashes (at least 500) and near-miss, aiming to form an effective training dataset. Following dataset construction, you can train the SafeDriver until the model's performance stabilizes, typically after about thirty cycles. Then, use the resultant model as a starting point for the subsequent training iteration, which again starts with data collection.

## 4. Ablation study
To further validate the impact of various data densification techniques within the dense learning algorithm, we provide code for conducting an ablation study. This study explores seven different variants of the dense learning algorithm, as outlined below:
- **No episodic data densification (NEDD)**: This variant omits the episodic data densification technique, meaning all trajectories are used to train the SafeDriver model. All other settings remain the same. For instance, the state densification technique still applies, training only on safety-critical states identified by neural network-based safety metrics.
- **No state-level data densification (NSLDD)**: This ablation evaluates the importance of state-level data densification by using all state information from selected crash and near-miss scenarios to train SafeDriver, rather than restricting to safety-critical states.
- **No near-miss episodes (NNME)**: In this experiment, all near-miss episodes are excluded while other techniques remain intact.
- **No retrospective data densification (NRDD)**: Here, the training dataset isn't reevaluated post-training step based on the evaluation results of the latest model, while other conditions stay consistent.
- **No near-miss episodes and retrospective data densification (NNME_NRDD)**: This variant involves a dataset consisting solely of crash data, without re-evaluation based on evaluation results after each training step.
- **No trajectory resampling by probability of occurrence in NDE (NRNDE)**: Diverging from the original method, this study randomly samples crash and near-miss scenarios, maintaining other techniques as is.
- **No reconnection of informative states in Markov process (NRSMDP)**: This experiment assesses the impact of reconnecting informative states in the Markov process. Instead of modifying it by removing non-informative states, it segments the original Markov process into separate chains when non-informative states are encountered.

**1. Data Preparation**: Begin by using the [Jupyter Notebook](../data_processing/processing_evaluation_data_ablationstudy.ipynb) to process the data in preparation for the ablation study.

**2. Configuration Update**: Modify the ``ablation_study_config`` parameter within the [default configuration file](../yaml_configs/training.yaml) as guided below:
```yaml
...
experiment_config:
  # configure the ablation study settings, please set this variable to be "None" if you don't want to do ablation study,
  # otherwise, please set this variable to be one of the following:
  # - "NEDD" represents "no episodic data densification", 
  # - "NSLDD" represents "no state-level data densification", 
  # - "NNME" represents "no near-miss episodes", 
  # - "NRDD" represents "no retrospective data densification", 
  # - "NNME_NRDD" represents no near-miss episodes and retrospective data densification", 
  # - "NRNDE" represents "no trajectory resampling by probability of occurrence in NDE", 
  # - "NRSMDP" represents "no reconnection of informative states in Markov process",
  ablation_study_config: "None"
  ...
```
**3. Run the Ablation Study**: Execute the following command to begin the study:
```bash
python main_ablationstudy.py
```

The resulting training outputs will maintain the same file structure as previously established. You can then export the trained models and assess their safety performance.

<- Last Page: [Preparing Dataset](prepare_dataset.md)

-> Next Page: [Demonstration](demonstration.md)