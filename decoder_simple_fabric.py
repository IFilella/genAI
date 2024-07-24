import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
from torchview import draw_graph
from torchinfo import summary
from transformers import AutoTokenizer

# Create a Tokenizer class
# This is an in-house tokenizer that combines two different tokenizers
# for protein sequences and SMILES strings

class Tokenizer:
    def __init__(self, prot_tokenizer_name='facebook/esm2_t33_650M_UR50D', mol_tokenizer_name='ibm/MolFormer-XL-both-10pct', delim='$'):
        self.prot_tokenizer = AutoTokenizer.from_pretrained(prot_tokenizer_name)
        self.mol_tokenizer = AutoTokenizer.from_pretrained(mol_tokenizer_name, trust_remote_code=True)
        self.delim = delim

    def tokenize_texts(self, prots, mols):
        tokenized_prots = self.prot_tokenizer(prots, padding='max_length', truncation=True, max_length=150, return_tensors='pt')
        tokenized_mols = self.mol_tokenizer(mols, padding='longest', return_tensors='pt')
        tokenized_delim = self.prot_tokenizer([self.delim] * len(prots), padding=True, return_tensors='pt')

        input_tensor = torch.cat((tokenized_prots['input_ids'], tokenized_delim['input_ids'], tokenized_mols['input_ids']), dim=1)
        vocab_size = self.prot_tokenizer.vocab_size + self.mol_tokenizer.vocab_size + 1
        
        return input_tensor, vocab_size
    
    def __call__(self, prots, mols):
        return self.tokenize_texts(prots, mols)
    
    
# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_layer, dropout, device):
        super(DecoderBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, ff_hidden_layer)
        self.linear2 = nn.Linear(ff_hidden_layer, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.device = device

    
    def forward(self, x,target_mask):
        target_mask = target_mask.to('cuda:%d' %self.device)
        x = x.to('cuda:%d' %self.device)
        attn_output, _ = self.self_attention(x, x, x, attn_mask=target_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x
    
# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = pe
        self.device=device
        #self.register_buffer("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x.to('cuda:%d'%self.device) + self.pe[:x.size(0), :].to('cuda:%d'%self.device)
        return self.dropout(x)
    
# We need to mask out input decoders to prevent attention to future positions
def generate_square_subsequent_mask(sz):
    """Generate a mask to prevent attention to future positions."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Multilayer Decoder
# This model has multiple decoder blocks stacked on top of each other
class MultiLayerTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers, device):
        super(MultiLayerTransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, device=device)
        self.transformer_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_hidden_layer, dropout, device)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.device = device


    def forward(self, x):
        x = x.long().clone()
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for transformer_block in self.transformer_blocks:
            target_mask = generate_square_subsequent_mask(x.size(0))
            target_mask = target_mask.to(self.device)
            x = transformer_block(x,target_mask)
        output = self.linear(x)
        output = self.softmax(output)
        return output


if __name__ == '__main__':
    
    
    mask = generate_square_subsequent_mask(sz=5)

    # TEST THE TOKENIZER CLASS
    
    tokenizer = Tokenizer()
    from data.fake_data import texts
    prot_seqs = [text.split('$')[0] for text in texts]
    smiles = [text.split('$')[1] for text in texts]

    input_tensor, vocab_size = tokenizer(prot_seqs, smiles)
    print(input_tensor.shape)
    
    
    # TEST THE MULTI LAYER TRANSFORMER DECODER
    
    # Follow the same process as before
    # Define the hyperparameters
    vocab_size     = 10000
    d_model        = 2048
    num_heads      = 2
    ff_hidden_layer  = 8*d_model
    dropout        = 0.1
    num_layers     = 12
    context_length = 1000
    batch_size     = 1

    # Create our input to the model to process
    input_tensor = torch.randint(0, vocab_size, (context_length, batch_size))

    # Initialize the model with `num_layer` layers
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers)
    