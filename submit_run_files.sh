#!/bin/bash

# Directory containing the run scripts
run_dir="runs/model_dimensions"

# Check if the directory exists
if [ ! -d "$run_dir" ]; then
  echo "Error: Directory $run_dir does not exist."
  exit 1
fi

# Loop through all the run scripts in the directory
for run_file in "$run_dir"/*.sh; do
  if [ -f "$run_file" ]; then
    echo "Submitting: $run_file"
    sbatch "$run_file"
  else
    echo "No run scripts found in $run_dir."
    exit 1
  fi
done

echo "All run scripts have been submitted."
