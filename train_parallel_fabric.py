import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
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

class ProtMolDataset(Dataset):
    def __init__(self, prot_seqs, smiles):
        self.prot_seqs = prot_seqs
        self.smiles = smiles
    def __len__(self):
        return len(self.prot_seqs)
    def __getitem__(self, idx):
        prot_seq = self.prot_seqs[idx]
        smile = self.smiles[idx]
        return prot_seq, smile
    
def collate_fn(batch, tokenizer):
    # Tokenize the protein sequences and SMILES strings
    prot_seqs = [prot_seq for prot_seq, _ in batch]
    smiles = [smile for _, smile in batch]
    prot_max_len = max(len(prot_seq) for prot_seq in prot_seqs)
    mol_max_len = max(len(smile) for smile in smiles)
    encoded_texts = tokenizer(prot_seqs, smiles, prot_max_length=prot_max_len, mol_max_length=mol_max_len)
    
    return {'input_ids': encoded_texts['input_ids'], 'attention_mask': encoded_texts['attention_mask']}

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

    torch.cuda.memory._record_memory_history(max_entries=100000)
    
    # Load the Dataset
    print('[Rank %d] Preparing the dataset...'%rank)
    dataset = ProtMolDataset(prot_seqs, smiles)
    
    # Split the dataset into training and validation sets
    print('[Rank %d] Splitting the dataset...'%rank)
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    if verbose:
        print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # Load the DataLoaders and Initialize the tokenizers
    print('[Rank %d] Initializing the tokenizers...'%rank)
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    
    print('[Rank %d] Initializing the dataloaders...'%rank)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=lambda x: collate_fn(x, tokenizer))

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=lambda x: collate_fn(x, tokenizer))
    
    # Model
    print('[Rank %d] Initializing the model...'%rank)
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers, device=rank)

    assert model.linear.out_features == vocab_size, f"Expected output layer size {combined_vocab_size}, but got {model.linear.out_features}"

    # Print model information
    if verbose:
        summary(model)

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

        for i, batch in enumerate(train_dataloader):
            
            input_tensor = batch['input_ids']
            input_att_mask = batch['attention_mask']

            # Generate the shifted input tensor for teacher forcing
            # Apply teacher forcing only after the delimiter token
            batch_size = input_tensor.size(0)
            input_tensor_shifted = input_tensor.clone()
            input_att_mask_shifted = input_att_mask.clone()
            
            if teacher_forcing:
                for i in range(batch_size):
                    delim_idx = (input_tensor[i] == tokenizer.combined_vocab['<DELIM>']).nonzero(as_tuple=True)
                    if len(delim_idx[0]) > 0:
                        start_idx = delim_idx[0].item() + 1
                        if start_idx < input_tensor.size(1):
                            input_tensor_shifted[i, start_idx:] = torch.cat([torch.zeros_like(input_tensor[i, start_idx:start_idx+1]), input_tensor[i, start_idx:-1]], dim=0)
                            input_att_mask_shifted[i, start_idx:] = torch.cat([torch.zeros_like(input_att_mask[i, start_idx:start_idx+1]), input_att_mask[i, start_idx:-1]], dim=0)
                input_tensor = input_tensor_shifted
                input_att_mask = input_att_mask_shifted
            else:
                input_tensor = input_tensor
                input_att_mask = input_att_mask

            input_tensor = input_tensor.detach()
            input_tensor = fabric.to_device(input_tensor)
            input_att_mask = fabric.to_device(input_att_mask)
            
            logits = model(input_tensor, input_att_mask, tokenizer.combined_vocab['<DELIM>'])

            # calculate the loss just for the second part (after the delimiter)
            # mask after the delimiter
            batch_size = input_tensor.size(0)
            loss_mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            for i in range(batch_size):
                delim_idx = (input_tensor[i] == tokenizer.combined_vocab['<DELIM>']).nonzero(as_tuple=True)
                if len(delim_idx[0]) > 0:
                    start_idx = delim_idx[0].item() + 1
                    loss_mask[i, start_idx:] = True
                    
            # Apply mask to the logits and labels
            logits = logits.view(batch_size, -1, vocab_size)
            logits = logits[loss_mask]
            labels = input_tensor[loss_mask]
            
            # Compute the loss
            loss = criterion(logits, labels)
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

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
                input_tensor = batch['input_ids']
                input_att_mask = batch['attention_mask']
                
                input_tensor = input_tensor.clone().detach()
                logits = model(input_tensor, input_att_mask, tokenizer.combined_vocab['<DELIM>'])

                # Mask after the delimiter
                batch_size = input_tensor.size(0)
                loss_mask = torch.zeros_like(input_tensor, dtype=torch.bool)
                for i in range(batch_size):
                    delim_idx = (input_tensor[i] == tokenizer.combined_vocab['<DELIM>']).nonzero(as_tuple=True)
                    if len(delim_idx[0]) > 0:
                        start_idx = delim_idx[0].item() + 1
                        loss_mask[i, start_idx:] = True
                
                logits = logits.view(batch_size, -1, vocab_size)
                logits = logits[loss_mask]
                labels = input_tensor[loss_mask]
                
                # Compute the loss
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
    
    if verbose:
        torch.cuda.memory._dump_snapshot('memory_snapshot.pickle')
    torch.cuda.memory._record_memory_history(enabled=None)

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
    verbose        = config['verbose']

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
            "dataset": "ChEMBL_BindingDB_sorted_sample10000",
            }
        )

    # Train the model
    train_model(prots, mols, prot_tokenizer_name, mol_tokenizer_name,
                num_epochs, learning_rate, batch_size, d_model, num_heads, ff_hidden_layer,
                dropout, num_layers, loss_function, optimizer, weights_path, get_wandb,
                teacher_forcing, validation_split, num_gpus, verbose)
    
    timef = time.time() - time0
    print('Time taken:', timef)


if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    main()


