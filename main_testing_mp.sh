#!/bin/bash

worker_number=1 # This variable defines the number of workers to be used in the parallel processing
yaml_path="./yaml_configs/testing.yaml" # Please replace "/path/to/your/yaml/configuration/file" with the path to your configuration file

while getopts n: flag
do
    case "${flag}" in
        n) worker_number=${OPTARG};;
    esac
done

for ((i = 0; i < worker_number; i++)); do
    echo "Worker: $i"
    python main_testing.py --yaml_path $yaml_path --worker_index $i & 
done