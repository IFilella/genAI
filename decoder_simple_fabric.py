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
from tokenizer import Tokenizer
from lightning.fabric import Fabric

fabric = Fabric(accelerator='cuda', devices=2, num_nodes=1)
fabric.launch()
rank = fabric.global_rank

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_layer, dropout):
        super(DecoderBlock, self).__init__()

        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, ff_hidden_layer)
        self.linear2 = nn.Linear(ff_hidden_layer, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, padding_mask, target_mask):
        x = fabric.to_device(x)
        target_mask = fabric.to_device(target_mask)
        padding_mask = fabric.to_device(padding_mask)
        padding_mask = padding_mask.to(dtype=torch.float32)
        target_mask = target_mask.to(dtype=torch.float32)
        
        attn_output, _ = self.self_attention(x, x, x, key_padding_mask=padding_mask, attn_mask=target_mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe
        
    def forward(self, x):
        x = fabric.to_device(x)
        self.pe = fabric.to_device(self.pe)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# We need to mask out input decoders to prevent attention to future positions
def generate_square_subsequent_mask(sz):
    """Generate a mask to prevent attention to future positions."""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Create a mask that starts masking after the token with DELIM ID
def create_partial_mask(sequence, token_id=33):
    """ Example output with this function:
        tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf],
                [0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
                [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf],
                [0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])"""

    seq_length = sequence.size(1)
    mask = torch.zeros(seq_length, seq_length)
    for i in range(sequence.size(0)):
        start_idx = (sequence[i] == token_id).nonzero(as_tuple=True)[0][0].item()
        if start_idx < seq_length:
            mask[start_idx:, start_idx:] = generate_square_subsequent_mask(seq_length - start_idx)
    return mask

def create_prefix_decoder_mask(sequence, token_id=33):
    """
    Create an attention mask for a prefix-decoder model.
    - The tokens before the `token_id` (prefix) can only attend to themselves.
    - The tokens after the prefix follow a standard subsequent mask.
    - The tokens before the prefix cannot attend to the tokens after the prefix.

    Args:
        sequence (torch.Tensor): Tensor of shape (batch_size, seq_len), containing tokenized sequences.
        token_id (int): The token id marking the boundary between prefix and subsequent tokens.

    Returns:
        torch.Tensor: The attention mask of shape (seq_len, seq_len).

        tensor([[0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
                [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
                [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
                [0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
                [0., 0., 0., 0., 0., -inf, -inf, -inf, -inf, -inf],
                [0., 0., 0., 0., 0., 0., -inf, -inf, -inf, -inf],
                [0., 0., 0., 0., 0., 0., 0., -inf, -inf, -inf],
                [0., 0., 0., 0., 0., 0., 0., 0., -inf, -inf],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., -inf],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    batch_size, seq_length = sequence.size()
    mask = torch.zeros(seq_length, seq_length)

    # Loop over each sequence in the batch
    for i in range(batch_size):
        # Find the index of the prefix token
        start_idx = (sequence[i] == token_id).nonzero(as_tuple=True)[0][0].item()

        # Apply subsequent mask for the tokens after the prefix
        if start_idx < seq_length:
            subsequent_mask = generate_square_subsequent_mask(seq_length - start_idx)
            mask[start_idx:, start_idx:] = subsequent_mask

        # Apply -inf to prevent the prefix (rows 0 to start_idx-1) from attending to tokens after the prefix (columns start_idx to seq_length-1)
        mask[:start_idx, start_idx:] = float('-inf')

    return mask

# Multilayer Decoder
# This model has multiple decoder blocks stacked on top of each other
class MultiLayerTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers):
        super(MultiLayerTransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_hidden_layer, dropout)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, padding_mask, delim_tokenidx):
        x = x.long().clone()
        x = fabric.to_device(x)
        padding_mask = fabric.to_device(padding_mask)

        target_mask = create_partial_mask(x, delim_tokenidx)

        x = self.embedding(x)
        print('Embedding shape:', x.shape)
        x = self.pos_encoder(x)
        print('Positional encoding shape:', x.shape)
        
        for transformer_block in self.transformer_blocks:
            # Generate a mask to prevent attention to future positions
            # We mask just the molecules (the second half of the input)
            target_mask = fabric.to_device(target_mask)
            x = transformer_block(x, padding_mask, target_mask)

        output = self.linear(x)
        output = self.softmax(output)
        return output


if __name__ == '__main__':


    # TEST THE TOKENIZER CLASS

    tokenizer = Tokenizer()
    from data.fake_data import texts
    prot_seqs = [text.split('$')[0] for text in texts]
    smiles = [text.split('$')[1] for text in texts]

    tokenized_sequences = tokenizer(prot_seqs, smiles)
    input_tensor = tokenized_sequences['input_ids']
    att_mask = tokenized_sequences['attention_mask']
    
    # Test masking functions
    """
    input_tensor = torch.tensor([[10, 39, 30, 25, 33, 15,  6,  5, 34,  9],
                                [37, 20,  1, 28, 33, 14, 45, 44, 48, 16],
                                [48,  8, 35, 37, 33, 38,  0, 35, 47, 11],
                                [36, 28, 18, 39, 33,  0, 34, 24, 30, 40],
                                [15, 32, 32, 21, 33,  1, 48, 40, 41, 31],
                                [45, 34, 27, 27, 33,  4, 38, 40, 37,  5],
                                [36, 47, 15, 45, 33,  0, 21, 44,  9, 16],
                                [27, 25, 39, 48, 33,  5, 11, 28,  8, 11],
                                [ 6, 41,  5, 42, 33, 21, 27, 14, 41,  7],
                                [35, 45, 22, 48, 33, 29, 27, 39,  8, 45]])

    print('Input tensor shape:', input_tensor.shape)
    print(input_tensor)

    delim_tokenidx = tokenizer.combined_vocab['<DELIM>']
    mask = create_partial_mask(input_tensor, delim_tokenidx)
    print('Mask shape:', mask.shape)
    print(mask)

    mask2 = create_prefix_decoder_mask(input_tensor, delim_tokenidx)
    print('Mask2 shape:', mask2.shape)
    print(mask2)
    """

    # TEST THE MULTI LAYER TRANSFORMER DECODER

    # Follow the same process as before
    # Define the hyperparameters
    vocab_size     = tokenizer.vocab_size
    d_model        = 2048
    num_heads      = 2
    ff_hidden_layer  = 8*d_model
    dropout        = 0.1
    num_layers     = 4
    context_length = 1000
    batch_size     = 1

    # Create our input to the model to process
    #input_tensor = torch.randint(0, vocab_size, (context_length, batch_size))
    input_tensor = fabric.to_device(input_tensor)
    att_mask = fabric.to_device(att_mask)
    print('Input tensor shape:', input_tensor.shape)
    print('Attention mask shape:', att_mask.shape)

    # Initialize the model with `num_layer` layers
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers)
    model = fabric.to_device(model)

    output = model(input_tensor, att_mask, tokenizer.delim_token_id)
    print(output.shape)

