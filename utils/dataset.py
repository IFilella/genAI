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

    input_ids = encoded_texts['input_ids']
    attention_mask = encoded_texts['attention_mask']

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
