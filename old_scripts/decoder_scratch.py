import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch import rand
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer
import torchvision
from torchview import draw_graph

class SimpleEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(SimpleEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = self.embedding(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create a positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional embeddings to input token embeddings
        x = x + self.pe[:, :x.size(1)]

        return x

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads (if not, Assertion Error is raised)"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_linear = torch.nn.Linear(d_model, d_model)
        self.key_linear = torch.nn.Linear(d_model, d_model)
        self.value_linear = torch.nn.Linear(d_model, d_model)

        self.concat_linear = torch.nn.Linear(d_model, d_model)


    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Linear projections for query, key, and value
        query = self.query_linear(x)  # Shape: [batch_size, seq_len, d_model]
        key = self.key_linear(x)  # Shape: [batch_size, seq_len, d_model]
        value = self.value_linear(x)  # Shape: [batch_size, seq_len, d_model]

        # Reshape query, key, and value to split into multiple heads
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Shape: [batch_size, num_heads, seq_len, head_dim]
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Shape: [batch_size, num_heads, seq_len, head_dim]
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Shape: [batch_size, num_heads, seq_len, head_dim]

        # Compute attention scores
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # Shape: [batch_size, num_heads, seq_len, seq_len]

        # Apply mask to prevent attending to future tokens
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # Shape: [batch_size, num_heads, seq_len, seq_len]

        # Weighted sum of value vectors based on attention weights
        context = torch.matmul(attention_weights, value)  # Shape: [batch_size, num_heads, seq_len, head_dim]

        # Reshape and concatenate attention heads
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # Shape: [batch_size, seq_len, num_heads * head_dim]
        output = self.concat_linear(context)  # Shape: [batch_size, seq_len, d_model]

        return output
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__() # d_ff is the number of neurons in the hidden layer
        self.linear1 = nn.Linear(d_model, d_ff) # Linear function
        self.linear2 = nn.Linear(d_ff, d_model) # Linear function

    def forward(self, x):
        x = F.relu(self.linear1(x)) # ReLU activation function for the hidden layer
        x = self.linear2(x) # Linear function for the output layer
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiheadSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffnn = FeedForwardNetwork(d_model, d_ff)
        
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffnn(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_len, vocab_size):
        super().__init__()
        self.embedding = SimpleEmbedding(vocab_size, d_model, max_len)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, max_len)
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.decoder(x) # This is the block that is repeated several times (now just 1 time)
        logits = self.linear(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


if __name__ == '__main__':
    
    # EXAMPLE USAGE
    
    # Example data  (THIS IS NOT REAL DATA!!)
    texts = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQLRCDQYP$CC(=O)OC1=CC=CC=C1C(=O)O",
        "AGADTPLAAIDRYGFQFQDIKHFTTSYQQLTTLNNEKQR$CC1=CC=CC=C1",
        "MKKLLFAIPLPVLTLAWLAPSSQAATAPAPAQPAPPQAA$NCC(=O)O"
    ]
    
    d_model = 500  # Dimension of the embeddings
    max_len = 100 # Maximum sequence length
    num_heads = 4  # Number of attention heads
    
    # 0. TOKENIZER
    print('Tokenizing the input sequences...')
    
        # Tokenize the input sequences
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '$'})
    tokenized_ids = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
        # Pad or truncate the sequences to the fixed max length
    padded_ids = [
        ids[:max_len] + [tokenizer.pad_token_id] * max(0, max_len - len(ids))
        for ids in tokenized_ids
    ]
    input_tensor = torch.tensor(padded_ids)
    vocab_size = tokenizer.vocab_size + 1
        # Embed the input tokens
    simple_embedding = SimpleEmbedding(vocab_size, d_model, max_len)
    input_embeddings = simple_embedding(input_tensor)

    
    # 1. POSITIONAL ENCODING
    print('\nAdding positional encoding to the input embeddings...')
    
    positional_encoding = PositionalEncoding(d_model, max_len)
    #input_token_embeddings = torch.randn(1, max_len, d_model) # this was an example
    output_embeddings = positional_encoding(input_embeddings)
    print(' Input token embeddings shape:', input_embeddings.shape)
    print(' Output positional embeddings shape:', output_embeddings.shape)


    """
    # 2. MASKED MULTI-HEAD ATTENTION
    # Masked self-attention prohibits us from looking forward in the sequence during self-attention
    print('\nCreating a causal mask...')
    
    num_heads = 4
    multihead_self_attention = MultiheadSelfAttention(d_model, num_heads)
    output_att = multihead_self_attention(output_embeddings)
    
    print(' Input embeddings shape:', output_embeddings.shape)
    print(' Output shape:', output_att.shape)
    
    # 3. RESIDUAL CONNECTIONS & LAYER NORMALIZATION

    print('\nAdding residual connections and layer normalization...')
    
    layernorm = nn.LayerNorm(d_model)
    output_normalized = layernorm(output_att + output_embeddings)
    
    print(' Input shape:', output_embeddings.shape)
    print(' Output shape:', output_normalized.shape)
    print(' Output mean:', output_normalized.mean().item())
    print(' Output std:', output_normalized.std().item())
    
    
    # 4. FEED FORWARD NETWORKS
    
    print('\nAdding feed-forward networks...')
    ffnn = FeedForwardNetwork(d_model, 2000)
    output_ffnn = ffnn(output_normalized)
    
    print(' Input shape:', output_normalized.shape)
    print(' Output shape:', output_ffnn.shape)
    
    
    # 5. RESIDUAL CONNECTIONS & LAYER NORMALIZATION
    
    print('\nAdding residual connections and layer normalization...')
    
    output_fnn_normalized = layernorm(output_ffnn + output_normalized)
    
    print(' Input shape:', output_normalized.shape)
    print(' Output shape:', output_fnn_normalized.shape)
    print(' Output mean:', output_fnn_normalized.mean().item())
    print(' Output std:', output_fnn_normalized.std().item())
    """
    # THIS COMMENTED PART SHOULD BE THE SAME AS THE FOLLOWING BLOCK
    
    # BLOCK: TRANSFORMER DECODER
    # Why this block? Because this block is repeated several times in a decoder-only model
    
    print('\nCreating a transformer decoder block...')

    decoder_block = TransformerDecoder(d_model, num_heads, 2000, max_len)
    output_decoder = decoder_block(output_embeddings)
    
    
    # 6. SOFTMAX OUTPUT LAYER
    
    print('\nAdding a softmax output layer...')
    
    linear_layer = nn.Linear(d_model, vocab_size)
    logits = linear_layer(output_decoder)
    
    probs = F.softmax(logits, dim=-1)
    print(' Output shape:', probs.shape)

    

    
    
    ###### ALL TOGETHER ######
    # we will use the Model class to train the model
    # so, we will only use this class with all the other ones implemented
    
    print('\nCreating the model...')
    model = Model(d_model, num_heads, 2000, max_len, vocab_size)
    logits, probs = model(input_tensor)
    print(model)
    print(logits.shape, probs.shape)
    
    model_graph = draw_graph(model, input_tensor, save_graph=True, expand_nested=True)