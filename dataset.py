import torch
from torch.utils.data import Dataset

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
    encoded_texts = tokenizer(prot_seqs, smiles)
    
    return {'input_ids': encoded_texts['input_ids'], 'attention_mask': encoded_texts['attention_mask']}
