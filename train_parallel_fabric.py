import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from decoder_simple_fabric import Tokenizer, MultiLayerTransformerDecoder
import pandas as pd
from lightning.fabric import Fabric
import argparse
import time
import wandb
import os
import yaml

def prepare_dataset(prot_seqs, smiles, prot_tokenizer_name, mol_tokenizer_name, batch_size=4):
    tokenizer = Tokenizer()
    input_tensor, vocab_size = tokenizer(prot_seqs, smiles)
    print('Input tensor shape:', input_tensor.shape)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(input_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, vocab_size


def train_model(prot_seqs,
                smiles,
                prot_tokenizer_name,
                mol_tokenizer_name,
                num_epochs=10,
                lr=0.0001,
                batch_size=4,
                d_model=1000,
                num_heads=8,
                ff_hidden_layer=4*1000,
                dropout=0.1,
                num_layers=12,
                loss_function='crossentropy',
                optimizer='Adam',
                weights_path='weights/best_model_weights.pth',
                get_wandb=False,
                teacher_forcing=False,
                num_gpus=2
                ):
    
    """
    Train the model using the specified hyperparameters.

    Args:
        prot_seqs (list): A list of protein sequences
        smiles (list): A list of SMILES strings
        prot_tokenizer_name (str): The name of the protein tokenizer to use
        mol_tokenizer_name (str): The name of the molecule tokenizer to use
        num_epochs (int, optional): The number of epochs to train the model. Defaults to 10.
        lr (float, optional): The learning rate. Defaults to 0.0001.
        batch_size (int, optional): The batch size. Defaults to 4.
        d_model (int, optional): The model dimension. Defaults to 1000.
        num_heads (int, optional): The number of attention heads. Defaults to 8.
        ff_hidden_layer (int, optional): The hidden layer size in the feedforward network. Defaults to 4*d_model.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        num_layers (int, optional): The number of transformer layers. Defaults to 12.
        loss_function (str, optional): The loss function to use. Defaults to 'crossentropy'.
        optimizer (str, optional): The optimizer to use. Defaults to 'Adam'.
        weights_path (str, optional): The path to save the model weights. Defaults to 'weights/best_model_weights.pth'.
        get_wandb (bool, optional): Whether to log metrics to wandb. Defaults to False.
        teacher_forcing (bool, optional): Whether to use teacher forcing. Defaults to False.
    
    """
        
    fabric = Fabric(devices=num_gpus)
    fabric.launch()
    
    rank = fabric.global_rank 
    print(rank)
    
    # prepare the dataset (distributed)
    print('Preparing the dataset...')
    dataloader, vocab_size = prepare_dataset(prot_seqs, smiles, prot_tokenizer_name, mol_tokenizer_name)
    
    # Model
    print('Initializing the model...')
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers, device=rank)
    
    
    # TO DO: Add support for other loss functions and optimizers
    # Loss function
    padding_token_id = 2 # Ensure that padding tokens are masked during training to prevent the model from learning to generate them.
    if loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss(ignore_index=padding_token_id)
    else:
        raise ValueError('Invalid loss function. Please use "crossentropy"')
    
    # Optimizer
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError('Invalid optimizer. Please use "Adam"')
    
    # Distribute the model to all available GPUs (using Fabric)
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    
    # Start the training loop
    best_accuracy = 0

    for epoch in range(num_epochs):
        
        model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            input_tensor = batch[0]
            
            # Generate the shifted input tensor for teacher forcing            
            if teacher_forcing:
                input_tensor_shifted = torch.cat([torch.zeros_like(input_tensor[:, :1]), input_tensor[:, :-1]], dim=1)
                input_tensor = input_tensor_shifted
            else:
                input_tensor = input_tensor
            
            optimizer.zero_grad()
            
            input_tensor = input_tensor.clone().detach()
            logits = model(input_tensor)

            # Reshape the logits and labels for loss calculation
            logits = logits.view(-1, vocab_size)
            labels = input_tensor.view(-1)
                
            loss = criterion(logits, labels)
        
            #loss.backward()
            fabric.backward(loss)
            optimizer.step()
        
            _, preds = torch.max(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()

            total_loss += loss.item()
            
        acc = total_correct / total_samples
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}, Accuracy: {acc}")

        if get_wandb:
            # log metrics to wandb
            wandb.log({"Epoch": epoch+1, "loss": total_loss, "accuracy": acc})
            
        # Save the model weights if accuracy improves
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), weights_path)
        
    print('Training complete!')
    
def main():
    
    time0 = time.time()
    
    parser = argparse.ArgumentParser(description='Train a Transformer Decoder model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file with all the parameters', required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Get the data (in this case, fake data)
    df = pd.read_csv(config['data_path'])
    df = df.sample(1000)
    prots = df[config['col_prots']].tolist()
    mols = df[config['col_mols']].tolist()
    
    # Define the hyperparameters
    d_model        = config['d_model']
    num_heads      = config['num_heads']
    ff_hidden_layer  = config['ff_hidden_layer']
    dropout        = config['dropout']
    num_layers     = config['num_layers']
    batch_size     = config['batch_size']
    num_epochs     = config['num_epochs']
    learning_rate  = config['learning_rate']
    loss_function  = config['loss_function']
    optimizer      = config['optimizer']
    weights_path   = config['weights_path']
    get_wandb      = config['get_wandb']
    num_gpus       = config['num_gpus']
    
    # Configure wandb
    if get_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=config['wandb_project'],

            # track hyperparameters and run metadata
            config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "d_model": d_model,
            "num_heads": num_heads,
            "ff_hidden_layer": ff_hidden_layer,
            "dropout": dropout,
            "num_layers": num_layers,
            "architecture": "Decoder-only",
            "dataset": "BindingDB_sample1000",
            }
        )
    
    
    # Train the model
    print('Training the model...')
    train_model(prots, mols, 'facebook/esm2_t33_650M_UR50', 'ibm/MolFormer-XL-both-10pct',
                num_epochs, learning_rate, batch_size, d_model, num_heads, ff_hidden_layer,
                dropout, num_layers, loss_function, optimizer, weights_path, get_wandb, num_gpus)
    
    timef = time.time() - time0
    print('Time taken:', timef)
    
    
if __name__ == '__main__':
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main()
    