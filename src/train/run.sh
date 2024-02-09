#!/bin/bash

# # Wait for pipeline configuration to be available 
until [ "$(ls -A "/mnt/remote/config/pipeline_config.json")" != "" ]
do   
  sleep 10
done
echo "pipeline configuration is available"

# echo "Running pipeline with configuration:"
# cat /mnt/remote/config/pipeline_config.json
# pytrain /mnt/remote/config/pipeline_config.json

echo "Running overlap analysis..."
python3.9 /mnt/remote/model/overlap.py