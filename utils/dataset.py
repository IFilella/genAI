import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split, DistributedSampler
import random
import numpy as np
import matplotlib.pyplot as plt

# DATASET CLASS
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

# COLLATE FUNCTION WITH TOKENIZER AND TEACHER FORCING IMPLEMENTED
def filter_sequences_by_unknown_tokens(input_ids, attention_mask, unknown_token_id, pad_token_id, threshold=0.2):
    
    unknown_token_count = (input_ids == unknown_token_id).sum(dim=1)
    total_token_count = (input_ids != pad_token_id).sum(dim=1)
    # the threshold is the ratio of unknown tokens to total tokens
    unknown_token_ratio = unknown_token_count / total_token_count
    valid_sequences_mask = unknown_token_ratio < threshold
    
    filtered_input_ids = input_ids[valid_sequences_mask]
    filtered_attention_mask = attention_mask[valid_sequences_mask]
    
    return {'input_ids': filtered_input_ids, 'attention_mask': filtered_attention_mask}

def collate_fn(batch, tokenizer, prot_max_length, mol_max_length):
    # Tokenize the protein sequences and SMILES strings
    prot_seqs = [prot_seq for prot_seq, _ in batch]
    smiles = [smile for _, smile in batch]

    encoded_texts = tokenizer(prot_seqs, smiles,
                              prot_max_length=prot_max_length,
                              mol_max_length=mol_max_length)

    encoded_texts = filter_sequences_by_unknown_tokens(encoded_texts['input_ids'],
                                                       encoded_texts['attention_mask'],
                                                       tokenizer.prot_tokenizer.unk_token_id,
                                                       tokenizer.prot_tokenizer.pad_token_id,
                                                       threshold=0.2)
    
    # teacher forcing by removing last element of input_ids and first element of labels
    input_ids = encoded_texts['input_ids'][:, :-1]
    attention_mask = encoded_texts['attention_mask'][:, :-1]
    labels = encoded_texts['input_ids'][:, 1:]

    # get labels with -100 (ignore_index from loss) to all protein tokenids
    protein_ids = tokenizer.prot_ids + [tokenizer.prot_tokenizer.pad_token_id] + [tokenizer.delim_token_id]
    labels = torch.where(torch.isin(labels, torch.tensor(protein_ids)), -100, labels)
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# DATA PREPARATION
def prepare_data(prot_seqs, smiles, validation_split, batch_size, tokenizer,
                 fabric, prot_max_length, mol_max_length, verbose, seed):
    """Prepares datasets, splits them, and returns the dataloaders."""
    
    if verbose >=0:
        fabric.print('Preparing the dataset...')
    dataset = ProtMolDataset(prot_seqs, smiles)

    if verbose >=0:
        fabric.print('Splitting the dataset...')
    
    # Split the dataset into train and validation randomly
    torch.manual_seed(seed)
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Sort the datasets based on the lengths of the sequences
    idxs_train = list(range(len(train_dataset)))
    idxs_val = list(range(len(val_dataset)))
    train_lengths = [(idx, len(train_dataset[idx][0]) + len(train_dataset[idx][1])) for idx in idxs_train]
    val_lengths = [(idx, len(val_dataset[idx][0]) + len(val_dataset[idx][1])) for idx in idxs_val]
    sorted_train_indices = [idx for idx, length in sorted(train_lengths, key=lambda x: x[1])]
    sorted_val_indices = [idx for idx, length in sorted(val_lengths, key=lambda x: x[1])]

    # Create the sorted datasets using the sorted indices
    train_dataset = Subset(train_dataset, sorted_train_indices)
    val_dataset = Subset(val_dataset, sorted_val_indices)
    
    # Check if the sort worked
    if verbose >= 2:
        fabric.print('Lenghts of the sequences in the train and validation datasets:')
        train_lengths = [len(train_dataset[idx][0]) + len(train_dataset[idx][1]) for idx in idxs_train]
        val_lengths = [len(val_dataset[idx][0]) + len(val_dataset[idx][1]) for idx in idxs_val]
        fabric.print(np.asarray(train_lengths))
        fabric.print(np.asarray(val_lengths))
    
    # Plot the distribution of the lengths of the proteins and molecules in the train and validation datasets
    if verbose >= 2:
        fabric.print('Plotting the distribution of the lengths of the proteins and molecules...')
        plt.figure()    
        train_prots_lenghts = [len(train_dataset[idx][0]) for idx in idxs_train]
        val_prots_lenghts = [len(val_dataset[idx][0]) for idx in idxs_val]
        plt.hist(train_prots_lenghts, bins=75, label='train proteins', alpha=0.5)
        plt.hist(val_prots_lenghts, bins=75, label='val proteins', alpha=0.5)
        plt.legend()
        plt.savefig('plots/hist_train_test_prots.pdf')
        
        plt.figure()
        train_mols_lenghts = [len(train_dataset[idx][1]) for idx in idxs_train]
        val_mols_lenghts = [len(val_dataset[idx][1]) for idx in idxs_val]
        plt.hist(train_mols_lenghts, bins=100, label='train mols',alpha=0.5)
        plt.hist(val_mols_lenghts, bins=100, label='val mols', alpha=0.5)
        plt.legend()
        plt.savefig('plots/hist_train_test_mols.pdf')
    
    if verbose >=1:
        fabric.print(f"Train dataset size: {len(train_dataset)}, "\
                    f"Validation dataset size: {len(val_dataset)}")
    
    if verbose >=0:
        fabric.print('Initializing the dataloaders...')
    
    sampler_train = DistributedSampler(train_dataset, shuffle=False, drop_last=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  sampler=sampler_train,
                                  collate_fn=lambda x: collate_fn(x, tokenizer,
                                                                  prot_max_length,
                                                                  mol_max_length))
    
    sampler_val = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=sampler_val,
                                collate_fn=lambda x: collate_fn(x, tokenizer,
                                                                prot_max_length,
                                                                mol_max_length))

    return train_dataloader, val_dataloader
