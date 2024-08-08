import torch
import torch.nn as nn
from transformers import AutoTokenizer
#from decoder import Model
from decoder_simple import MultiLayerTransformerDecoder
from tokenizer import Tokenizer, MolecularTokenizer
import torch.nn.functional as F
import pandas as pd
import time


# Generate new text (SMILES strings) using the model
def generate_smiles(model, sequence, delim='$', max_length=50, temperature=1.0, device='cuda'):
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Tokenize the protein sequence
    tokenizer = Tokenizer()
    tokenizer.load_combined_vocab('combined_vocab.json')
    
    # tokenize the protein sequence according to the loaded combined vocab
    tokenized_prot = []
    max_len = 450
    seq = sequence + delim
    prot_ids = [tokenizer.token2id.get(char, tokenizer.token2id.get('<unk>')) for char in seq]
    prot_ids = prot_ids[:max_len] + [tokenizer.token2id.get('<pad>')] * (max_len - len(prot_ids))
    tokenized_prot.append(prot_ids)
    input_tensor = torch.tensor(tokenized_prot)
    input_tensor = input_tensor.to(device)
    #print('Input tensor shape: ', input_tensor.shape)
    
    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_length):
            # Predict the next token
            logits = model(input_tensor)
            
            # next_token_logits = logits[:, -1, :]  # Get logits for the last token in the sequence
            # next_token_id = torch.argmax(next_token_logits, dim=1).item() # Get the predicted token id
                        
            next_token_logits = logits[:, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1).item()
            
            # Check if there is any out-of-vocabulary token
            if next_token_id >= len(tokenizer.combined_vocab):
                next_token_id = tokenizer.combined_vocab['<unk>']
            
            # Stop generation if the end token is generated
            if next_token_id == tokenizer.combined_vocab['<eos>']:
                break
            
            # Append the generated token id to the list
            generated_tokens.append(next_token_id)
            next_token_tensor = torch.tensor([[next_token_id]])
            next_token_tensor = next_token_tensor.to(device)
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)

    # Decode the generated token ids
    #print(generated_tokens)
    
    generated_smiles = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_smiles

def generate(quantity, sequence, delim='$', max_length=50, temperature=1.0, device='cuda'):
    for i in range(quantity):
        print(f'Generating SMILES string {i+1}...')
        generated_smiles = generate_smiles(model, sequence, delim, max_length, temperature, device)
        print(generated_smiles)
        print('\n')
    


if __name__ == '__main__':
    
    time0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the hyperparameters --> make sure that they are the same as the ones in the trained model!!
    d_model        = 1024
    num_heads      = 8
    ff_hidden_layer  = 4*d_model
    dropout        = 0.1
    num_layers     = 4
    batch_size     = 100
    num_epochs     = 10
    learning_rate  = 0.0001
    weights_path   = 'weights/model_weights_test1-1.pth'

    # Define the vocabulary size from the combined vocab file (also from training!)
    vocab_size = 71
    
    # Instantiate the model
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers, device)
    model = model.to(device)
    
    # Load the model weights
    print('Loading model weights...')
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    # Generate SMILES strings using the trained model given a protein sequence
    print('Generating SMILES strings...')
    sequence = 'MEQPQEEAPEVREEEEKEEVAEAEGAPELNGGPQHALPSSSYTDLSRSSSPPSLLDQLQMGCDGASCGSLNMECRVCGDKASGFHYGVHACEGCKGFFRRTIRMKLEYEKCERSCKIQKKNRNKCQYCRFQKCLALGMSHNAIRFGRMPEAEKRKLVAGLTANEGSQYNPQVADLKAFSKHIYNAYLKNFNMTKKKARSILTGKASHTAPFVIHDIETLWQAEKGLVWKQLVNGLPPYKEISVHVFYRCQCTTVETVRELTEFAKSIPSFSSLFLNDQVTLLKYGVHEAIFAMLASIVNKDGLLVANGSGFVTREFLRSLRKPFSDIIEPKFEFAVKFNALELDDSDLALFIAAIILCGDRPGLMNVPRVEAIQDTILRALEFHLQANHPDAQYLFPKLLQKMADLRQLVTEHAQMMQRIKKTETETSLHPLLQEIYKDMY'
    print(len(sequence))
    generated_smiles = generate_smiles(model, sequence,
                                        delim='$', max_length=100,
                                        temperature=2.0, device=device)
    print(generated_smiles)
    
    some_generated_smiles = generate(10, sequence, delim='$', max_length=100, temperature=1.0, device=device)
    print(some_generated_smiles)