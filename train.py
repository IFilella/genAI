import torch
import torch.nn as nn
import torch.optim as optim
from decoder_model import MultiLayerTransformerDecoder
from utils.dataset import prepare_data
from utils.earlystopping import EarlyStopping
from utils.configuration import load_config
from utils import memory
from tokenizer import Tokenizer
from sklearn.metrics import precision_score, recall_score, f1_score
from lightning.fabric import Fabric
from torchinfo import summary
import pandas as pd
import argparse
import time
import wandb


def train_epoch(model, dataloader, criterion, optimizer, tokenizer, fabric):
    """Train the model for one epoch."""

    model.train()

    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0

    for batch in dataloader:
        input_tensor = batch['input_ids']
        input_att_mask = batch['attention_mask']
        labels = batch['labels']
        logits = model(input_tensor, input_att_mask, tokenizer.delim_token_id, fabric)
        
        # Flatten logits first two dimensions (concatenate seqs from batch)
        logits = logits.contiguous().view(-1, logits.size(-1))
        
        # Flatten masked_labels dimensions (concatenate seqs from batch)
        labels = labels.contiguous().view(-1)

        # Compute the loss
        loss = criterion(logits, labels)
        total_train_loss += loss.item()

        # Backward pass and optimization
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True) # set_to_none=True to save memory

        # Calculate the accuracy
        _, preds = torch.max(logits, dim=1)
        mask = (labels != -100)
        labels_mol = labels[mask]
        preds_mol = preds[mask]
        total_train_correct += (preds_mol == labels_mol).sum().item()
        total_train_samples += labels.numel()

    avg_train_loss = total_train_loss / len(dataloader)
    train_acc = total_train_correct / total_train_samples

    return avg_train_loss, train_acc

def evaluate_epoch(model, dataloader, criterion, tokenizer, vocab_size, fabric):
    """Evaluate the model for one epoch."""

    model.eval()

    total_val_loss = 0
    total_val_correct = 0
    total_val_samples = 0

    with torch.no_grad():

        for batch in dataloader:

            input_tensor = batch['input_ids']
            input_att_mask = batch['attention_mask']
            labels = batch['labels']

            logits = model(input_tensor, input_att_mask, tokenizer.delim_token_id, fabric)

            # Flatten logits and labels dimensions (concatenate seqs from batch)
            logits = logits.contiguous().view(-1, logits.size(-1))
            labels = labels.contiguous().view(-1)

            # Compute the loss
            loss = criterion(logits, labels)
            total_val_loss += loss.item()
            
            # Calculate the accuracy
            _, preds = torch.max(logits, dim=1)
            total_val_correct += (preds == labels).sum().item()
            total_val_samples += labels.numel()

    avg_val_loss = total_val_loss / len(dataloader)
    val_acc = total_val_correct / total_val_samples

    # Calculate additional metrics
    precision = precision_score(labels.cpu(), preds.cpu(),
                                average='macro', zero_division=0)
    recall = recall_score(labels.cpu(), preds.cpu(),
                          average='macro', zero_division=0)
    f1 = f1_score(labels.cpu(), preds.cpu(),
                    average='macro', zero_division=0)

    other_metrics = {'precision': precision, 'recall': recall, 'f1': f1}

    return avg_val_loss, val_acc, other_metrics

# TRAINING FUNCTION
def train_model(prot_seqs,
                smiles,
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
                wandb_project=None,
                wandb_config=None,
                validation_split = 0.2,
                num_gpus=2,
                verbose=False,
                prot_max_length=600,
                mol_max_length=80,
                patience=5,
                delta=0
                ):

    """
    Train the model using the specified hyperparameters.

    Args:
        prot_seqs (list): A list of protein sequences
        smiles (list): A list of SMILES strings
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
        wandb_project (str, optional): The wandb project name. Defaults to None.
        wandb_config (dict, optional): The wandb configuration dictionary. Defaults to None.
        validation_split (float, optional): The fraction of the data to use for validation. Defaults to 0.2.
        num_gpus (int, optional): The number of GPUs to use. Defaults to 2.
        verbose (bool, optional): Whether to print model information. Defaults to False.
        prot_max_length (int, optional): The maximum length of the protein sequences. Defaults to 600.
        mol_max_length (int, optional): The maximum length of the SMILES strings. Defaults to 80.
        patience (int, optional): The number of epochs to wait before early stopping. Defaults to 5.
        delta (int, optional): The minimum change in validation loss to qualify as an improvement. Defaults to 0.
    """
    fabric = Fabric(accelerator='cuda', devices=num_gpus, num_nodes=1, strategy='ddp', precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(1234)
    rank = fabric.global_rank
    
    if get_wandb:
        if fabric.is_global_zero:
            wandb.init(
                project=wandb_project,
                config=wandb_config
            )

    # Enable memory tracking
    # torch.cuda.memory._record_memory_history(max_entries=100000)

    # Tokenizer initialization
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size

    # Data preparation
    train_dataloader, val_dataloader = prepare_data(prot_seqs, smiles,
                                                    validation_split, batch_size,
                                                    tokenizer, fabric, prot_max_length,
                                                    mol_max_length, verbose)

    # Model
    if fabric.is_global_zero:
        print('Initializing the model...')
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads,
                                         ff_hidden_layer, dropout, num_layers)
    model = fabric.to_device(model)

    assert model.linear.out_features == vocab_size,\
    f"Expected output layer size {vocab_size}, but got {model.linear.out_features}"

    # Print model information
    if verbose:
        if fabric.is_global_zero:
            summary(model)

    # TO DO: Add support for other loss functions and optimizers
    # Loss function
    if loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
    else:
        raise ValueError('Invalid loss function. Please use "crossentropy"')

    # Optimizer
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError('Invalid optimizer. Please use "Adam"')

    # Distribute the model using Fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = fabric.setup_dataloaders(val_dataloader)

    # Start the training loop
    if fabric.is_global_zero:
        print('Starting the training loop...')
    early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=verbose)
    for epoch in range(num_epochs):

        # training
        avg_train_loss, train_acc = train_epoch(model, train_dataloader,
                                                criterion, optimizer,
                                                tokenizer, fabric)
        exit()
        avg_train_acc = fabric.all_reduce(train_acc, reduce_op='mean')
        
        # validation
        avg_val_loss, val_acc, other_metrics = evaluate_epoch(model, val_dataloader,
                                                              criterion, tokenizer,
                                                              vocab_size, fabric)
        avg_val_acc = fabric.all_reduce(val_acc, reduce_op='mean')

        # Print the metrics
        if fabric.is_global_zero:
            print(f"Epoch {epoch+1}/{num_epochs}, "\
                f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}, "\
                f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc:.4f}")

            if verbose:
                print(f"Precision: {other_metrics['precision']:.4f}, "\
                    f"Recall: {other_metrics['recall']:.4f}, F1: {other_metrics['f1']:.4f}")

            if get_wandb:
                # log metrics to wandb
                wandb.log({"Epoch": epoch+1, "Train Loss": avg_train_loss,
                        "Train Accuracy": avg_train_acc,
                        "Validation Loss": avg_val_loss,
                        "Validation Accuracy": avg_val_acc,
                        "Validation Precision": other_metrics['precision'],
                        "Validation Recall": other_metrics['recall'],
                        "Validation F1": other_metrics['f1']})

        # Early stopping based on validation loss
        early_stopping(avg_val_loss, model, weights_path)

        if early_stopping.early_stop:
            print(f"Early stopping after {epoch+1} epochs.")
            break
        
        if verbose:
            if fabric.global_rank == 0: 
                memory.get_GPU_memory(device='cuda:0')
                memory.get_CPU_memory()
            if fabric.global_rank == 1:
                memory.get_GPU_memory(device='cuda:1')

    print('[Rank %d] Training complete!'%rank)

    # Disable memory tracking
    # if verbose:
    #     torch.cuda.memory._dump_snapshot('memory_snapshot.pickle')
    # torch.cuda.memory._record_memory_history(enabled=None)


def parse_args():
    """Parse the command-line arguments."""

    parser = argparse.ArgumentParser(description='Train a Transformer Decoder model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration YAML file with all the parameters',
                        required=True)
    return parser.parse_args()

def main():
    torch.set_float32_matmul_precision('medium')

    time0 = time.time()

    args = parse_args()

    config = load_config(args.config)

    # Get the data (in this case, it is sampled for testing purposes)
    df = pd.read_csv(config['data_path'])
    df = df.sample(1000)
    prots = df[config['col_prots']].tolist()
    mols = df[config['col_mols']].tolist()

    # Train the model
    train_model(prots, mols, config['num_epochs'], config['learning_rate'],
                config['batch_size'], config['d_model'], config['num_heads'],
                config['ff_hidden_layer'], config['dropout'], config['num_layers'],
                config['loss_function'], config['optimizer'], config['weights_path'],
                config['get_wandb'], config['wandb']['wandb_project'], config['wandb']['wandb_config'],
                config['validation_split'], config['num_gpus'], config['verbose'],
                config['prot_max_length'], config['mol_max_length'],
                config['es_patience'], config['es_delta'])

    timef = time.time() - time0
    print('Time taken:', timef)


if __name__ == '__main__':

    main()
