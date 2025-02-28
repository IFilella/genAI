
from lightning.fabric import Fabric
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generate import generate
from decoder_model import MultiLayerTransformerDecoder
from tokenizer import Tokenizer

# List of protein sequences --> taken from the cluster representatives (top 100 most populated clusters)
df = pd.read_csv('test_generation/cluster_representatives_100.csv', index_col=0)
seqs = df['Sequence'].tolist()

weights_files = ['../weights/weights_dm256_nh4_ff1024_nl20.pth',\
                '../weights/weights_dm512_nh8_ff4096_nl4.pth', \
                '../weights/weights_dm768_nh12_ff4608_nl2.pth']

results = []

for i, weight in enumerate(weights_files):
    
    print(i, weight)
    
    # Define the hyperparameters --> take them from the weights name
    d_model        = int(weight.split('_')[1].split('dm')[1])
    num_heads      = int(weight.split('_')[2].split('nh')[1])
    ff_hidden_layer  = int(weight.split('_')[3].split('ff')[1])
    num_layers     = int(weight.split('_')[4].split('nl')[1].split('.')[0])
    dropout       = 0.25
    print(d_model, num_heads, ff_hidden_layer, num_layers)
    
    # Initialize the model
    tokenizer = Tokenizer()
    vocab_size = tokenizer.vocab_size
    
    model = MultiLayerTransformerDecoder(vocab_size, d_model, num_heads, ff_hidden_layer, dropout, num_layers)
    
    # Load the model weights
    fabric = Fabric(accelerator='cuda', devices=1)
    fabric.launch()
    state = {'model': model}
    fabric.load(weight, state)
    
    # Generate SMILES strings for each protein sequence
    for j, seq in enumerate(seqs):
        
        generated_smiles, sucess_rate = generate(100, seq, model, fabric, max_length=80,\
                                                temperature=1.2, verbose=False,\
                                                outdir='test_generation',\
                                                outname='generated_molecules_weights_%d_seq_%d.csv'%(i, j))
        print(i, j, sucess_rate)
        
        results.append([i, weight, j, seq, sucess_rate])
        
df_results = pd.DataFrame(results, columns=['weight_idx', 'weight', 'seq_idx', 'sequence', 'success_rate'])
df_results.to_csv('test_generation/results_success_rate.csv', index=False)