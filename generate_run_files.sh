#!/bin/bash

# Create directory to store the run scripts
mkdir -p runs/model

# Path to the generated YAML configuration files
config_dir="configs/model"

# Loop through all the YAML files in the config directory
for config_file in "$config_dir"/*.yaml; do

  # Extract the base name of the configuration file (e.g., config_1.yaml)
  config_base=$(basename "$config_file")
  config_base=${config_base%.yaml}
  
  # Generate the run script
  run_file="runs/model/run_${config_base}.sh"
  
  cat <<EOL > "$run_file"
#!/bin/bash
#SBATCH --job-name=train_${counter}
#SBATCH --account=bsc72
#SBATCH --chdir=.
#SBATCH --output=logs/train_${counter}_%j.out
#SBATCH --error=logs/train_${counter}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --qos=acc_bscls

module purge

ml bsc/1.0
ml anaconda
source activate genAI

wandb login 7952be2b9b469177c60dcaee07f53602e3f2f7f3
wandb offline

srun python -u train.py --config $config_file
EOL

  echo "Generated: $run_file"

done

echo "All run scripts have been generated in the 'runs' directory."