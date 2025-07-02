import pandas as pd
import urllib, urllib.request
from io import StringIO
import numpy as np
import multiprocessing as mp


def map_sequences(input_csv, mapping_tsv, output_csv):
    # Load your input DataFrame (with Uniprot IDs)
    df = pd.read_csv(input_csv)

    # Load the large TSV mapping file
    mapping_df = pd.read_csv(mapping_tsv, sep='\t', usecols=['Entry', 'Sequence'])
    mapping_df.rename(columns={'Entry': 'Uniprot'}, inplace=True)

    # Merge on the Uniprot column
    merged_df = df.merge(mapping_df, on='Uniprot', how='left')

    # Save the output with sequences
    merged_df.to_csv(output_csv, index=False)

    print(f"Sequence mapping complete. Output saved to {output_csv}")
    
def map_sequences_efficient(input_csv, mapping_tsv, output_csv, chunk_size=500000):
    # Load the input CSV with UniProt IDs
    df = pd.read_csv(input_csv)

    # Extract unique Uniprot IDs from your input
    uniprot_ids = set(df['Uniprot'].unique())

    # Initialize empty list to collect matched chunks
    matched_chunks = []

    # Read the large mapping file in chunks
    for chunk in pd.read_csv(mapping_tsv, sep='\t', usecols=['Entry', 'Sequence'], chunksize=chunk_size):
        # Filter only rows where Entry is in your input Uniprot IDs
        filtered_chunk = chunk[chunk['Entry'].isin(uniprot_ids)].copy()
        filtered_chunk.rename(columns={'Entry': 'Uniprot'}, inplace=True)
        matched_chunks.append(filtered_chunk)

    # Concatenate all matched rows
    mapping_df = pd.concat(matched_chunks, ignore_index=True)

    # Merge with the input DataFrame
    merged_df = df.merge(mapping_df, on='Uniprot', how='left')

    # Save the output
    merged_df.to_csv(output_csv, index=False)

    print(f"Sequence mapping complete. Output saved to {output_csv}")

if __name__ == '__main__':

    input_csv = '../data/raw/SMPBind_part_16.csv'
    output_csv = '../data/raw/SMPBind_seq_part_16.csv'
    mapping_tsv = '../data/raw/uniprot_seqs.tsv'

    #map_sequences(input_csv, mapping_tsv, output_csv) # this process is being killed due to memory issues
    map_sequences_efficient(input_csv, mapping_tsv, output_csv, chunk_size=1000)  # Adjust chunk size as needed
