import torch
import torch.nn as nn
from decoder_model import MultiLayerTransformerDecoder
from tokenizer import Tokenizer, ProteinTokenizer
import torch.nn.functional as F
import pandas as pd
from lightning.fabric import Fabric
import time
from utils.molecular_properties import compute_properties


# Generate new text (SMILES strings) using the model
def generate_smiles(model, sequence, max_length=50, temperature=1.0, verbose=True):
    
    fabric = Fabric(accelerator='cuda', devices=1)
    fabric.launch()
    
    model.eval()  # Set the model to evaluation mode
    model = fabric.to_device(model)

    # Tokenize the protein sequence and add the delimiter token
    prot_tokenizer = ProteinTokenizer()
    tokenized_prot = prot_tokenizer(sequence)
    prot_ids, prot_att_mask = tokenized_prot['input_ids'], tokenized_prot['attention_mask']
    
    tokenizer = Tokenizer()
    delim_id = torch.tensor([tokenizer.delim_token_id]).unsqueeze(0)
    delim_att_mask = torch.tensor([0]).unsqueeze(0)
    
    input_tensor = torch.cat([prot_ids[0], delim_id[0]], dim=0).unsqueeze(0)
    input_att_mask = torch.cat([prot_att_mask[0], delim_att_mask[0]], dim=0).unsqueeze(0)
    
    if verbose:
        print('Input tensor shape:', input_tensor.shape)
        print('Attention mask shape:', input_att_mask.shape)
    
    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_length):
            
            # Predict the next token
            logits = model(input_tensor, input_att_mask, tokenizer.delim_token_id, fabric)
                        
            next_token_logits = logits[:, -1, :] / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1).item()
            
            # Check if there is any out-of-vocabulary token
            if next_token_id >= tokenizer.vocab_size:
                next_token_id = prot_tokenizer.unk_token_id
            
            # Stop generation if the end token is generated
            if next_token_id == prot_tokenizer.eos_token_id:
                break
            
            # Append the generated token id to the list
            generated_tokens.append(next_token_id)
            next_token_tensor = torch.tensor([[next_token_id]])
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
            input_att_mask = torch.cat([input_att_mask, torch.tensor([[0]])], dim=1)

    # Decode the generated token ids
    if verbose:
        print('Generated tokens:', generated_tokens)
        
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def generate(quantity, sequence, max_length=50, temperature=1.0, verbose=False, outdir='.'):
    valid_smiles = []
    for i in range(quantity):
        generated_smiles = generate_smiles(model, sequence, max_length, temperature, verbose=verbose)
        print(generated_smiles)
        try:
            smi, sa, qed, mw, logp, tpsa, nhd, nha = compute_properties(generated_smiles)
            valid_smiles.append([smi, sa, qed, mw, logp, tpsa, nhd, nha])
        except:
            pass
        
    # Save the properties in a dataframe
    df_smiles_props = pd.DataFrame(valid_smiles, columns=['smiles', 'SAscore','QED', 'mw', 'logp', 'tpsa', 'nHD', 'nHA'])
    df_smiles_props.to_csv('%s/generated_smiles.csv'%outdir, index=False)
    
    return df_smiles_props

if __name__ == '__main__':
    
    time0 = time.time()

    # Define the hyperparameters --> make sure that they are the same as the ones in the trained model!!
    d_model        = 1024
    num_heads      = 8
    ff_hidden_layer  = 4*d_model
    dropout        = 0.1
    num_layers     = 2
    batch_size     = 32
    num_epochs     = 10
    learning_rate  = 0.0001
    weights_path   = 'weights/best_model_weights.pth'

    # Define the vocabulary size
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    
    # Instantiate the model
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers)
    
    # Load the model weights
    print('Loading model weights...')
    state_dict = torch.load(weights_path, map_location='cuda')
    model.load_state_dict(state_dict, strict=False)

    # Generate SMILES strings using the trained model given a protein sequence
    print('Generating SMILES strings...')
    sequence = 'MEQPQEEAPEVREEEEKEEVAEAEGAPELNGGPQHALPSSSYTDLSRSSSPPSLLDQLQMGCDGASCGSLNMECRVCGDKASGFHYGVHACEGCKGFFRRTIRMKLEYEKCERSCKIQKKNRNKCQYCRFQKCLALGMSHNAIRFGRMPEAEKRKLVAGLTANEGSQYNPQVADLKAFSKHIYNAYLKNFNMTKKKARSILTGKASHTAPFVIHDIETLWQAEKGLVWKQLVNGLPPYKEISVHVFYRCQCTTVETVRELTEFAKSIPSFSSLFLNDQVTLLKYGVHEAIFAMLASIVNKDGLLVANGSGFVTREFLRSLRKPFSDIIEPKFEFAVKFNALELDDSDLALFIAAIILCGDRPGLMNVPRVEAIQDTILRALEFHLQANHPDAQYLFPKLLQKMADLRQLVTEHAQMMQRIKKTETETSLHPLLQEIYKDMY'
    print(len(sequence))
    
    generated_smiles = generate_smiles(model, sequence, max_length=100, temperature=1.0)
    print(generated_smiles)
    
    some_generated_smiles = generate(100, sequence, max_length=100, 
                                     temperature=1.0, verbose=False)
    print(some_generated_smiles)
    print('Generated SMILES:', len(some_generated_smiles))