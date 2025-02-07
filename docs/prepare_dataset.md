# Preparing Dataset
After collecting the raw data from evaluation, please use the [Jupyter Notebook](../data_processing/processing_evaluation_data.ipynb) to process the data and get prepared for the training process. 

> Remember to change the variables named "**root_folder**" and "**exp_name**" in the second cell to be the absolute path to your data folder and the experiment name of your evaluation.

Please press the "Run All" button to process the data. The whole process might need hours based on the size of the raw data. The processed data will be stored in the folder "/absolute/path/to/data/folder/Experiment-testing_DateOfRunning/densified_exps", which should have the following structure: 
```
/absolute/path/to/data/folder/Experiment-testing_DateOfRunning/densified_exps/
|__crash/ # the folder contains the processed crash data
|__tested_and_safe/ # the folder contains the processed safe data
|__offline_av_alldata.json # the file contains the raw information of all the data
|__offline_av_alldata_new.json # the file contains the processed information of all the data
|__offline_av_nearmiss_new.json # the file contains the processed information of all the near-miss data
|__offline_av_neweval_crashnearmiss_new_origin.json # the original file contains the processed information of all the crash and near-miss data
|__offline_av_neweval_crashnearmiss_new.json # the latest file contains the processed information of all the crash and near-miss data
``` 

<- Last Page: [Data Collection/Evaluation](data_collection_evaluation.md)

-> Next Page: [Training](training.md)