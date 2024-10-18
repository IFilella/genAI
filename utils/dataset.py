import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

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
def filter_sequences_by_unknown_tokens(input_ids, attention_mask, unknown_token_id, threshold=0.2):
    
    unknown_token_count = (input_ids == unknown_token_id).sum(dim=1)
    total_tokens = input_ids.sum(dim=1)

    # the threshold is the ratio of unknown tokens to total tokens
    unknown_token_ratio = unknown_token_count / total_tokens
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
                                                       threshold=0.2)

    # teacher forcing by removing last element of input_ids and first element of labels
    input_ids = encoded_texts['input_ids'][:, :-1]
    attention_mask = encoded_texts['attention_mask'][:, :-1]
    labels = encoded_texts['input_ids'][:, 1:]

    # get labels with -100 (ignore_index from loss) to all protein tokenids
    protein_ids = set(tokenizer.prot_tokenizer.id2token.keys())
    special_ids = set([tokenizer.prot_tokenizer.cls_token_id,
                      tokenizer.prot_tokenizer.eos_token_id,
                      tokenizer.prot_tokenizer.unk_token_id])
    protein_ids = list(protein_ids - special_ids)
    protein_ids.append(tokenizer.delim_token_id)
    labels = torch.where(torch.isin(labels, torch.tensor(protein_ids)), -100, labels)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# DATA PREPARATION
def prepare_data(prot_seqs, smiles, validation_split, batch_size, tokenizer,
                 fabric, prot_max_length, mol_max_length, verbose):
    """Prepares datasets, splits them, and returns the dataloaders."""
    
    if fabric.is_global_zero:
        print('Preparing the dataset...')
    dataset = ProtMolDataset(prot_seqs, smiles)

    if fabric.is_global_zero:
        print('Splitting the dataset...')
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    if verbose:
        if fabric.is_global_zero:
            print(f"Train dataset size: {len(train_dataset)}, "\
                  f"Validation dataset size: {len(val_dataset)}")

    if fabric.is_global_zero:
        print('Initializing the dataloaders...')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  collate_fn=lambda x: collate_fn(x, tokenizer,
                                                                  prot_max_length,
                                                                  mol_max_length))

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=lambda x: collate_fn(x, tokenizer,
                                                                prot_max_length,
                                                                mol_max_length))

    return train_dataloader, val_dataloader
