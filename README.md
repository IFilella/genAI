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

## Model Architecture
**GenAI** uses a prefix-decoder architecture based on a decoder-only Transformer. This architecture is designed for generation tasks and excels at creating molecules when given a specific protein target. The prefix, which encodes information about the target protein, guides the decoding process, making the generated molecules highly specific to the target protein.

I will attach a plot with the architecture soon!

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

## Contributing
Contributions are welcome! Please submit issues or pull requests for enhancements or bug fixes.

## License
This project is licensed under the x License. See LICENSE for more details.

##

This README provides a structured overview of the project, its setup, and its usage. Let me know if you need any specific details added.