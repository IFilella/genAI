#!/bin/bash

# Define ranges for parameters
d_model=(512 1024 2048)                    # dimension of the model
num_heads=(8 16 32)                        # num of heads in the multi-head self-attention
ff_hidden_layer=(2048 4096 8192)           # dimension of the feed-forward hidden layer
num_layers=(4 6 8)                         # number of decoder blocks

dropout=(0.2)                      # percentage of dropout
learning_rate=(0.0001)             # learning rates
weight_decay=(0.001)               # weight decay in the L2 regularization (adam/adamW optimizer)
betas=("0.9,0.999")                # betas parameter in adamW optimizer

# Create directory to store the generated YAML files
mkdir -p configs/model_dimensions

# Loop through the parameter ranges
for dm in "${d_model[@]}"; do
  for nh in "${num_heads[@]}"; do
    for ff in "${ff_hidden_layer[@]}"; do
      for nl in "${num_layers[@]}"; do
        for do in "${dropout[@]}"; do
          for lr in "${learning_rate[@]}"; do
            for wd in "${weight_decay[@]}"; do
              for b in "${betas[@]}"; do
            
                # Generate the YAML filename
                filename="config_dm${dm}_nh${nh}_ff${ff}_nl${nl}"
                # Remove the decimal point from the filename
                filename=$(echo $filename | tr -d .)
                
                # Create the YAML content
                cat <<EOL > configs/model_dimensions/$filename.yaml

data_path: 'data/data_ChEMBL_BindingDB_clean.csv'
col_prots: 'Sequence'
col_mols: 'SMILES'
d_model: $dm
num_heads: $nh
ff_hidden_layer: $ffh
dropout: $do
num_layers: $nl
batch_size: 64
num_epochs: 12
learning_rate: $lr
loss_function: 'crossentropy'
optimizer: 'AdamW'
weight_decay: $wd
betas: [$b]
weights_path: "weights/weights_dm${dm}_nh${nh}_ff${ff}_nl${nl}"
validation_split: 0.2
get_wandb: true
wandb_project: 'train_decoder_hyperparameters'
wandb_name: 'decoder'
num_gpus: 4
verbose: 2
prot_max_length: 600
mol_max_length: 80
es_patience: 6
es_delta: 0.0001

EOL

                echo "Generated: configs/model_dimensions/$filename.yaml"

              done
            done
          done
        done
      done
    done
  done
done

echo "YAML configuration files generated in the configs directory."
