import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from decoder_simple_fabric import MultiLayerTransformerDecoder
from tokenizer import Tokenizer
import pandas as pd
from lightning.fabric import Fabric
import argparse
import time
import wandb
import os
import yaml
from torchinfo import summary

# TRAINING FUNCTION
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
                validation_split = 0.2,
                num_gpus=2,
                verbose=False
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
        validation_split (float, optional): The fraction of the data to use for validation. Defaults to 0.2.
        num_gpus (int, optional): The number of GPUs to use. Defaults to 2.
        verbose (bool, optional): Whether to print model information. Defaults to False.
    """
        
    fabric = Fabric(accelerator='cuda', devices=num_gpus, num_nodes=1)
    fabric.launch()
    
    rank = fabric.global_rank 
    print(rank)
    
    # prepare the dataset (distributed)
    print('[Rank %d] Preparing the dataset...'%rank)
    
    # Here I create the vocab file beacuse I need to have the same vocab length for all the processes
    if fabric.global_rank == 0:
        tokenizer = Tokenizer()
        tokenizer.build_combined_vocab(prot_seqs, smiles)
        tokenizer.save_combined_vocab('combined_vocab.json')
    fabric.barrier()
    
    # With the previously vocab file, I can load the vocab and create the input tensor
    tokenizer = Tokenizer()
    tokenizer.load_combined_vocab('combined_vocab.json')
    input_tensor, vocab_size = tokenizer(prot_seqs, smiles, use_loaded_vocab=True)
    
    # Split the dataset into training and validation sets
    print('[Rank %d] Splitting the dataset...'%rank)
    dataset = TensorDataset(input_tensor)
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    print('[Rank %d] Initializing the model...'%rank)
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers, device=rank)
    
    assert model.linear.out_features == vocab_size, f"Expected output layer size {combined_vocab_size}, but got {model.linear.out_features}"
    
    # Print model information
    if verbose:
        summary(model, (1000, batch_size), col_names=["input_size", "output_size", "num_params"])
    
    # TO DO: Add support for other loss functions and optimizers
    # Loss function
    padding_token_id = tokenizer.combined_vocab['<pad>'] # Ensure that padding tokens are masked during training to prevent the model from learning to generate them.
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
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    
    # Start the training loop
    print('[Rank %d] Starting the training loop...'%rank)
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        
        model.train()
        
        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0
        
        for batch in train_dataloader:
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
            total_train_correct += (preds == labels).sum().item()
            total_train_samples += labels.numel()

            total_train_loss += loss.item()
            
        train_acc = total_train_correct / total_train_samples
            
        print(f"[Rank {rank}] Epoch {epoch+1}/{num_epochs}, Train Loss: {total_train_loss}, Train Accuracy: {train_acc}")

        # validation
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_tensor = batch[0]
                input_tensor = input_tensor.clone().detach()
                logits = model(input_tensor)

                # Reshape the logits and labels for loss calculation
                logits = logits.view(-1, vocab_size)
                labels = input_tensor.view(-1)
                
                loss = criterion(logits, labels)
                
                _, preds = torch.max(logits, dim=1)
                total_val_correct += (preds == labels).sum().item()
                total_val_samples += labels.numel()

                total_val_loss += loss.item()
                
        val_acc = total_val_correct / total_val_samples
        
        print(f"[Rank {rank}] Epoch {epoch+1}/{num_epochs}, Validation Loss: {total_val_loss}, Validation Accuracy: {val_acc}")
        
        if get_wandb:
            # log metrics to wandb
            wandb.log({"Epoch": epoch+1, "Train Loss": total_train_loss, "Train Accuracy": train_acc,
                        "Validation Loss": total_val_loss, "Validation Accuracy": val_acc})
            
        # Save the model weights if validation accuracy improves
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), weights_path)
        
    print('[Rank %d] Training complete!'%rank)
    
def main():
    
    time0 = time.time()
    
    parser = argparse.ArgumentParser(description='Train a Transformer Decoder model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file with all the parameters', required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Get the data (in this case, it is sampled)
    df = pd.read_csv(config['data_path'])
    df = df.sample(1000)
    prots = df[config['col_prots']].tolist()
    mols = df[config['col_mols']].tolist()
    prot_tokenizer_name = config['protein_tokenizer']
    mol_tokenizer_name = config['smiles_tokenizer']
    
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
    teacher_forcing = config['teacher_forcing']
    validation_split = config['validation_split']
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
            "dataset": "BindingDB_sample10000",
            }
        )
    
    # Train the model
    train_model(prots, mols, prot_tokenizer_name, mol_tokenizer_name,
                num_epochs, learning_rate, batch_size, d_model, num_heads, ff_hidden_layer,
                dropout, num_layers, loss_function, optimizer, weights_path, get_wandb,
                teacher_forcing, validation_split, num_gpus, verbose=True)
    
    timef = time.time() - time0
    print('Time taken:', timef)
    
    
if __name__ == '__main__':
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main()
    