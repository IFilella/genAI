import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from decoder_simple import Tokenizer, MultiLayerTransformerDecoder
import pandas as pd
import argparse
import time
import os

def prepare_dataset(prot_seqs, smiles, prot_tokenizer_name, mol_tokenizer_name, batch_size=4):
    tokenizer = Tokenizer()
    input_tensor, vocab_size = tokenizer(prot_seqs, smiles)
    print('Input tensor shape:', input_tensor.shape)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(input_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, vocab_size


def train_model(rank,
                world_size,
                prot_seqs,
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
                ):
    
    """
    Train the model using the specified hyperparameters.

    Args:
        rank (int): The rank of the process
        world_size (int): The number of processes
        prot_seqs (list): List of protein sequences
        smiles (list): List of SMILES strings
        prot_tokenizer_name (str): The name of the protein tokenizer
        mol_tokenizer_name (str): The name of the molecule tokenizer
        num_epochs (int, optional): The number of epochs to train the model. Defaults to 10.
        lr (float, optional): The learning rate. Defaults to 0.0001.
        batch_size (int, optional): The batch size. Defaults to 4.
        d_model (int, optional): The model dimension. Defaults to 1000.
        num_heads (int, optional): The number of heads in the multi-head attention. Defaults to 8.
        ff_hidden_layer (int, optional): The number of hidden units in the feedforward layer. Defaults to 4*d_model.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        num_layers (int, optional): The number of layers in the transformer. Defaults to 12.
        loss_function (str, optional): The loss function to use. Defaults to 'crossentropy'.
        optimizer (str, optional): The optimizer to use. Defaults to 'Adam'.
        weights_path (str, optional): The path to save the model weights. Defaults to 'weights/best_model_weights.pth'.
        get_wandb (bool, optional): Whether to log metrics to wandb. Defaults to False.
        teacher_forcing (bool, optional): Whether to use teacher forcing. Defaults to False.
    
    """
    # Initialize the process group for distributed training
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    #device = torch.device(f'cuda:{rank}')
    #print(f'Running on rank {rank} with device {device}')
    
    # prepare the dataset (distributed)
    print('Preparing the dataset...')
    dataloader, vocab_size = prepare_dataset(prot_seqs, smiles, prot_tokenizer_name, mol_tokenizer_name)

    sampler = DistributedSampler(dataloader.dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataloader.dataset, batch_size=4, sampler=sampler)
    
    # Model
    # Distribute the model to all available GPUs
    print('Initializing the model...')
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers, device=rank)
    model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    #### TO DO: Add support for other loss functions and optimizers!!! ####
    
    # Loss function
    padding_token_id = 2 # Ensure that padding tokens are masked during training to prevent the model from learning to generate them.
    # Be aware that if change the tokenizer, the padding token id may change
    
    if loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss(ignore_index=padding_token_id)
    else:
        raise ValueError('Invalid loss function. Please use "crossentropy"')
    
    # Optimizer
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError('Invalid optimizer. Please use "Adam"')
    
    # Start the training loop
    print('Starting the training loop...')
    
    best_accuracy = 0
    for epoch in range(num_epochs):
                
        model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            input_tensor = batch[0].to(rank)
            
            # Generate the shifted input tensor for teacher forcing            
            if teacher_forcing:
                input_tensor_shifted = torch.cat([torch.zeros_like(input_tensor[:, :1]), input_tensor[:, :-1]], dim=1).to(rank)
                input_tensor = input_tensor_shifted
            else:
                input_tensor = input_tensor.to(rank)
            
            input_tensor = input_tensor.detach().clone().to(rank)
            
            optimizer.zero_grad()
            
            logits = model(input_tensor)

            # Reshape the logits and labels for loss calculation
            logits = logits.view(-1, vocab_size)
            labels = input_tensor.view(-1)
                
            loss = criterion(logits, labels)
        
            loss.backward()
            optimizer.step()
        
            _, preds = torch.max(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()

            total_loss += loss.item()
            
        acc = total_correct / total_samples
            
        print(f"Rank {rank}, Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}, Accuracy: {acc}")

        if get_wandb:
            # log metrics to wandb
            wandb.log({"Epoch": epoch+1, "loss": total_loss, "accuracy": acc})
            
        # Save the model weights if accuracy improves
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), weights_path)
    
    dist.destroy_process_group()
    print('Training complete!')
    
def main():
    
    time0 = time.time()
    
    # Get the data (in this case, fake data)
    df = pd.read_csv('data/data_seqmol_BindingDB_filt.csv', index_col=0)
    df = df.sample(10)

    # Define data for training
    prots = [str(i) for i in df['Sequence'].tolist()]
    mols = [str(i) for i in df['SMILES'].tolist()]
    prot_tokenizer_name = 'facebook/esm2_t33_650M_UR50'
    mol_tokenizer_name = 'ibm/MolFormer-XL-both-10pct'

    # Define the hyperparameters for training
    d_model        = 1000
    num_heads      = 8
    ff_hidden_layer  = 4*d_model
    dropout        = 0.1
    num_layers     = 12
    batch_size     = 8
    num_epochs     = 10
    learning_rate  = 0.0001
    loss_function  = 'crossentropy'
    optimizer      = 'Adam'
    weights_path   = 'weights/best_model_weights.pth'
    get_wandb      = False
    
    # Configure wandb
    if get_wandb:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="train_decoder_simple",

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
    
    
    # Train the model parallelly
    
    
    print('Training the model...')
    world_size = torch.cuda.device_count()
    print(world_size)
    
    mp.spawn(train_model,
            args=(world_size, prots, mols, prot_tokenizer_name, mol_tokenizer_name, 
                  num_epochs, learning_rate, batch_size, d_model, num_heads, ff_hidden_layer,
                  dropout, num_layers, loss_function, optimizer, weights_path, get_wandb),
            nprocs=world_size,
            join=True)
    
    # processes = []
    # mp.set_start_method(method='spawn')
    # for rank in range(world_size):
    #     p = mp.Process(target=train_model, args=(rank, world_size, prots, mols, prot_tokenizer_name, mol_tokenizer_name, 
    #                 num_epochs, learning_rate, batch_size, d_model, num_heads, ff_hidden_layer,
    #                 dropout, num_layers, loss_function, optimizer, weights_path, get_wandb))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    
    timef = time.time() - time0
    print('Time taken:', timef)
    
    
if __name__ == '__main__':
    
    # set environment variables for distributed training
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '22'
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    main()
    