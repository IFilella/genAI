<a id="readme-top"></a>

# genAI💊

## Introduction
**GenAI** is a generative model designed to accelerate drug discovery by creating novel molecules targeted at specific proteins.
Using a **prefix-decoder architecture** (decoder-only transformer), GenAI can generate molecular structures with potential binding affinity for specified protein targets.

## Table of Contents
- [Intruduction](#introduction) ✨
- [Model Architecture](#model-architecture) 🚀
- [Getting Started](#getting-started) 🛠️
- [Repository Structure](#repository-structure) 🔍
- [Usage](#usage) 📖
- [Configuration](#configuration) ⚙️
- [Contributing](#contributing) 🤝
- [License](#license) 📜
- [Contact](#contact) 📩
- [Acknowledgments](#acknowledgments) 👥

## Model Architecture
**GenAI** uses a prefix-decoder architecture based on a decoder-only Transformer. This architecture is designed for generation tasks and excels at creating molecules when given a specific protein target. The prefix, which encodes information about the target protein, guides the decoding process, making the generated molecules highly specific to the target protein.

I will attach a plot with the architecture soon!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started
1. **Clone the Repository**:
```bash
git clone https://github.com/juliavilmor/genAI
cd genAI
```
2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Structure
```bash
genAI
├── data/                   # Example datasets and datased used for training
├── data_preparation/       # Data preparation scripts
├── old_scripts/            # --> I will remove this folder from the repo
├── utils/                  # Utils scripts
├── wandb/                  # wandb tracking files
├── weights/                # Weights from training the model
├── examples/               # --> TO DO
├── config.yaml             # Configuration file with parameters for training
├── decoder_model.py        # The prefix-decoder model
├── generate.py             # Script for generate molecules
├── README.md               # Project documentation
├── requirements.txt        # Project requirements
├── run_training.sh         # Script for run the training in a cluster
├── tokenizer.py            # The tokenizer used in the model
└── train.py                # Training script
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage
1. **Train the model**:
```bash
python train.py --config config.yaml
```
or **Load a pretrained model** from:
```bash
'weights/model_weights.pth'
```
2. **Generate Molecules**:
```bash
python generate.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Configuration
Example of config file for training:

```yaml
data_path: 'data/data_ChEMBL_BindingDB_clean.csv'    # dataset path for training
col_prots: 'Sequence'                                # Column name of the protein sequences
col_mols: 'SMILES'                                   # Column name of the molecular SMILES
d_model: 512                                         # Dimension of the model (embedding)
num_heads: 8                                         # Num of heads in the multi-head self-attention
ff_hidden_layer: 2048                                # Dimension of the feed-forward hidden layer
num_layers: 4                                        # Number of decoder blocks
dropout: 0.2                                         # Percentage of dropout
batch_size: 64                                       # Batch size
num_epochs: 10                                       # Number of epochs to train
learning_rate: 0.0001                                # Learning rate
loss_function: 'crossentropy'                        # Loss function type
optimizer: 'AdamW'                                   # Optimizer type
weight_decay: 0.01                                   # Weight decay in the L2 regularization (adam/adamW optimizer)
betas: [0.9,0.999]                                   # Betas parameter in adamW optimizer
weights_path: 'weights/model_weights_test2-10.pth'   # Path to store the training weights
validation_split: 0.2                                # Split percentage for train and validation datasets
get_wandb: true                                      # Track the results with Weights&Biases
wandb_project: 'train_decoder_parallel_test2'        # wandb project name
wandb_name: 'decoder'                                # wandb name for tracking the job
num_gpus: 4                                          # Number of GPUs used for training
verbose: 2                                           # Level of verbosity: 0, 1, 2
prot_max_length: 600                                 # Maximum length of the protein sequences
mol_max_length: 80                                   # Maximun length of the molecules
es_patience: 6                                       # Patience parameter for early stopping
es_delta: 0.0001                                     # Delta parameter for early stopping
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing
Contributions are welcome! Please submit issues or pull requests for enhancements or bug fixes.

**Top contributors:**

<a href="https://github.com/juliavilmor/genAI/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=juliavilmor/genAI" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License
This project is licensed under the x License. See LICENSE for more details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contact
Júlia Vilalta-Mor - julia.vilalta@bsc.es   |   Isaac Filella-Mercè - isaac.filella1@bsc.es   |   Víctor Guallar - victor.guallar@bsc.es

Project Link: [https://github.com/juliavilmor/genAI](https://github.com/juliavilmor/genAI)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Acknowledgments
ARTIBAND

<p align="right">(<a href="#readme-top">back to top</a>)</p>


##

This README provides a structured overview of the project, its setup, and its usage. Let me know if you need any specific details added.