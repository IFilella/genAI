import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import wandb
from decoder_simple import Tokenizer, MultiLayerTransformerDecoder
import pandas as pd
import argparse
import time

def prepare_dataset(prot_seqs, smiles, prot_tokenizer_name, mol_tokenizer_name, batch_size=4):
    tokenizer = Tokenizer()
    input_tensor, vocab_size = tokenizer(prot_seqs, smiles)
    print('Input tensor shape:', input_tensor.shape)

    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(input_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, vocab_size


def train_model(model,
                dataloader,
                vocab_size, 
                num_epochs=10,
                lr=0.0001,
                batch_size=4,
                loss_function='crossentropy',
                optimizer='Adam',
                weights_path='weights/best_model_weights.pth',
                get_wandb=False,
                teacher_forcing=False):
    
    """
    Train the model using the specified hyperparameters.

    Args:
        model (nn.Module): The model to be trained
        dataloader (DataLoader): The DataLoader object containing the dataset
        vocab_size (int): The size of the vocabulary
        num_epochs (int, optional): The number of epochs to train the model. Defaults to 10.
        lr (float, optional): The learning rate. Defaults to 0.0001.
        batch_size (int, optional): The batch size. Defaults to 4.
        loss_function (str, optional): The loss function to use. Defaults to 'crossentropy'.
        optimizer (str, optional): The optimizer to use. Defaults to 'Adam'.
        weights_path (str, optional): The path to save the model weights. Defaults to 'weights/best_model_weights.pth'.
        get_wandb (bool, optional): Whether to log metrics to wandb. Defaults to False.
        teacher_forcing (bool, optional): Whether to use teacher forcing. Defaults to False.
    
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # Start the training loop
    best_accuracy = 0

    for epoch in range(num_epochs):
        
        # Distribute the model to all available GPUs
        model = nn.DataParallel(model) # so, wrap the model in DataParallel
        model.to(device)
        
        model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in dataloader:
            input_tensor = batch[0]
            input_tensor = input_tensor.to(device)

            # Generate the shifted input tensor for teacher forcing            
            if teacher_forcing:
                input_tensor_shifted = torch.cat([torch.zeros_like(input_tensor[:, :1]), input_tensor[:, :-1]], dim=1)
                input_tensor = input_tensor_shifted
            else:
                input_tensor = input_tensor
            
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get the data (in this case, fake data)
    df = pd.read_csv('data/data_seqmol_BindingDB_filt.csv')
    df = df.sample(1000)
    prots = df['Sequence'].tolist()
    mols = df['SMILES'].tolist()
    
    # Define the hyperparameters
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
    
    # Prepare the dataset
    print('Preparing the dataset...')
    dataloader, vocab_size = prepare_dataset(prots, mols, 'facebook/esm2_t33_650M_UR50', 'ibm/MolFormer-XL-both-10pct', batch_size=batch_size)
    print(vocab_size)
    
    # Initialize the model
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers, device)
    
    # Train the model
    print('Training the model...')
    train_model(model,
                dataloader,
                vocab_size, 
                num_epochs=num_epochs,
                lr=learning_rate,
                batch_size=batch_size,
                loss_function=loss_function,
                optimizer=optimizer,
                weights_path=weights_path,
                get_wandb=get_wandb,
                teacher_forcing=False)
    
    timef = time.time() - time0
    print('Time taken:', timef)
    
    
if __name__ == '__main__':
    
    main()
    