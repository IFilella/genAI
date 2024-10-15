import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

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
    
def collate_fn(batch, tokenizer, prot_max_length, mol_max_length):
    # Tokenize the protein sequences and SMILES strings
    prot_seqs = [prot_seq for prot_seq, _ in batch]
    smiles = [smile for _, smile in batch]
    encoded_texts = tokenizer(prot_seqs, smiles,
                              prot_max_length=prot_max_length,
                              mol_max_length=mol_max_length)
    
    return {'input_ids': encoded_texts['input_ids'], 'attention_mask': encoded_texts['attention_mask']}

# DATA PREPARATION
def prepare_data(prot_seqs, smiles, validation_split, batch_size, tokenizer,
                 rank, prot_max_length, mol_max_length, verbose):
    """Prepares datasets, splits them, and returns the dataloaders."""

    print('[Rank %d] Preparing the dataset...'%rank)
    dataset = ProtMolDataset(prot_seqs, smiles)

    print('[Rank %d] Splitting the dataset...'%rank)
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    if verbose:
        print(f"[Rank {rank}] Train dataset size: {len(train_dataset)}, "\
              f"Validation dataset size: {len(val_dataset)}")

    print('[Rank %d] Initializing the dataloaders...'%rank)
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
