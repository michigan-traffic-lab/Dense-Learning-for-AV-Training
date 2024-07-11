# Preparing Dataset
After collecting the raw data from evaluation, please use the [Jupyter Notebook](data_processing/processing_evaluation_data.ipynb) to process the data and get prepared for the training process. Please change the variables named "root_folder" and "exp_name" in the second cell to the path and experiment name of your evaluation results. Then, please press the "Run All" button to process the data. The whole process might need hours based on the size of the raw data. The processed data will be stored in the folder "Dense-Learning-for-AV-Training/output/Experiment-testing_DateOfRunning/densified_exps", which has the following structure: 
```
Dense-Learning-for-AV-Training/output/Experiment-testing_DateOfRunning/densified_exps/
|__crash/
|__tested_and_safe/
|__offline_av_alldata.json
|__offline_av_alldata_new.json
|__offline_av_nearmiss_new.json
|__offline_av_neweval_crashnearmiss_new.json
``` 

<- Last Page: [Data Collection/Evaluation](data_collection_evaluation.md)

-> Next Page: [Training](training.md)