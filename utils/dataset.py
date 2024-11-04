import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import random

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

# Custom function to randomly split but keep original order in each subset
def ordered_random_split(dataset, split_ratio=0.8, seed=0):
    random.seed(seed)
    total_size = len(dataset)
    
    train_size = int(total_size * split_ratio)
    test_size = total_size - train_size
    
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # Split random indices into train and test and sort them to keep order
    train_indices = sorted(indices[:train_size])
    test_indices = sorted(indices[train_size:])
    
    # Create Subsets using sorted indices
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset

# DATA PREPARATION
def prepare_data(prot_seqs, smiles, validation_split, batch_size, tokenizer,
                 fabric, prot_max_length, mol_max_length, verbose):
    """Prepares datasets, splits them, and returns the dataloaders."""
    
    if verbose >=0 and fabric.is_global_zero:
        print('Preparing the dataset...')
    dataset = ProtMolDataset(prot_seqs, smiles)

    if verbose >=0 and fabric.is_global_zero:
        print('Splitting the dataset...')

    train_dataset, val_dataset = ordered_random_split(dataset, split_ratio=validation_split, seed=0)
    
    if verbose >=1 and fabric.is_global_zero:
            print(f"Train dataset size: {len(train_dataset)}, "\
                  f"Validation dataset size: {len(val_dataset)}")

    if verbose >=0 and fabric.is_global_zero:
        print('Initializing the dataloaders...')
    
    sampler_train = DistributedSampler(train_dataset, shuffle=False)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  sampler=sampler_train,
                                  collate_fn=lambda x: collate_fn(x, tokenizer,
                                                                  prot_max_length,
                                                                  mol_max_length))
    
    sampler_val = DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                sampler=sampler_val,
                                collate_fn=lambda x: collate_fn(x, tokenizer,
                                                                prot_max_length,
                                                                mol_max_length))

    return train_dataloader, val_dataloader
