import pandas as pd
import urllib, urllib.request
from io import StringIO
import numpy as np
import multiprocessing as mp

df = pd.read_csv('data/raw/chembl_all.csv')
print(len(df))

def get_protein_sequences(uniprot_id):
    """Retrieves the sequence from the UniProt database based on the UniProt id."""
    line = urllib.parse.quote(uniprot_id)
    url = f'https://rest.uniprot.org/uniprotkb/search?query={line}&format=fasta'
    with urllib.request.urlopen(url) as f:
        fasta = f.read().decode('utf-8').strip()
        
    # get just the sequence
    fasta_seq = ''.join(fasta.split('\n')[1:])
    print(uniprot_id, fasta_seq)
    
    return fasta_seq

def process_chunk(chunk):
    chunk['Sequence'] = chunk['Target_uniprot'].apply(get_protein_sequences)
    return chunk

# Function to parallelize the apply
def parallelize_dataframe(df, func, num_partitions):
    # Split the dataframe into smaller chunks
    df_split = np.array_split(df, num_partitions)
    
    # Create a pool of workers
    pool = mp.Pool(num_partitions)
    
    # Apply the function in parallel on each chunk
    df = pd.concat(pool.map(func, df_split))
    
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()
    
    return df
    
def process_in_batches(file_path, chunksize, num_partitions):
    chunks = pd.read_csv(file_path, chunksize=chunksize)

    result_df = pd.DataFrame()  # Initialize an empty dataframe to store the results

    for chunk in chunks:
        processed_chunk = parallelize_dataframe(chunk, process_chunk, num_partitions)
        result_df = pd.concat([result_df, processed_chunk], ignore_index=True)
        # Optionally, you could save the result_df to a file after each iteration to free memory

    return result_df


# parallelize because there is a lot of data
num_partitions = 10
chunksize = 1000
df = process_in_batches('data/raw/chembl_all.csv', chunksize, num_partitions)

df.to_csv('data/raw/chembl_all_seq.csv', index=False)
df.to_pickle('data/raw/chembl_all_seq.pkl')
