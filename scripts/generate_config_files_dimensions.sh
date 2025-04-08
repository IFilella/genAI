#!/bin/bash

# Create directory to store the YAML configuration files
mkdir -p configs/model_dimensions

# Define parameter values from the table
d_model=(256 256 512 512 512 512 640 640 512 512 768 1024 1024 768)
ff_hidden_layer=(1024 1536 2048 2048 2048 2048 2560 2560 4096 4096 3072 4096 4096 4608)
num_layers=(20 16 8 8 6 6 4 4 4 4 3 2 2 2)
num_heads=(4 4 4 8 4 8 4 8 4 8 12 8 12 12)

# Loop through the parameters and generate configuration files
for i in "${!d_model[@]}"; do
  # Generate the YAML filename
  filename="config_dm${d_model[$i]}_nh${num_heads[$i]}_ff${ff_hidden_layer[$i]}_nl${num_layers[$i]}"
  # Remove the decimal point from the filename
  filename=$(echo $filename | tr -d .)

  # Create the YAML content
  cat <<EOL > configs/model_dimensions/$filename.yaml

data_path: 'data/data_ChEMBL_BindingDB_Plinder_clean.csv'
col_prots: 'Sequence'
col_mols: 'SMILES'
d_model: ${d_model[$i]}
num_heads: ${num_heads[$i]}
ff_hidden_layer: ${ff_hidden_layer[$i]}
dropout: 0.25
num_layers: ${num_layers[$i]}
batch_size: 64
num_epochs: 12
learning_rate: 0.0001
loss_function: 'crossentropy'
optimizer: 'AdamW'
weight_decay: 0.001
betas: [0.9,0.99]
weights_path: 'weights/weights_dm${d_model[$i]}_nh${num_heads[$i]}_ff${ff_hidden_layer[$i]}_nl${num_layers[$i]}'
validation_split: 0.2
get_wandb: True
wandb_project: 'train_decoder_hyperparameters'
wandb_name: 'decoder'
num_gpus: 4
verbose: 2
prot_max_length: 600
mol_max_length: 80
es_patience: 6
es_delta: 0.0001
seed: 1234
checkpoint_epoch: False
resume_training: False

EOL

  echo "Generated: configs/model_dimensions/$filename.yaml"
done

echo "All YAML configuration files have been created in 'configs/model_dimensions'."