from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import multiprocessing as mp
import os
from collections import Counter

# Load the raw data
df = pd.read_csv('../data/raw/chembl_all_seq.csv', index_col=0)
print(df)

# Drop rows with empty SMILES or target sequence
df = df[df['SMILES'].notna()]
df = df[df['Sequence'].notna()]
print(len(df))

# Filter rows with proteins no longer than 540 amino acids and larger than 60 amino acids
df = df[df['Sequence'].map(lambda x: len(str(x)) < 540)]
df = df[df['Sequence'].map(lambda x: len(str(x)) > 60)]
print(len(df))

# Filter molecules with SMILES length smaller than 80 and larger than 4
df = df[df['SMILES'].map(lambda x: len(str(x)) < 80)]
df = df[df['SMILES'].map(lambda x: len(str(x)) > 4)]
print(len(df))

# Just in case, fix the protein sequences that are written like chunks
df['Sequence'] = df['Sequence'].apply(lambda x: x.replace(' ','').replace('\n',''))

# Uppercase the protein sequences
df['Sequence'] = df['Sequence'].apply(lambda x: x.upper())

# Curate the SMILES: saninitize the molecules
from curate_dataset_mols_prots import sanitize_molecules
df['SMILES'] = df['SMILES'].apply(sanitize_molecules)
df = df.dropna(subset=['SMILES'])
print(len(df))

# Curate the protein Sequences: remove non-canonical amino acids
df['Sequence'] = df['Sequence'].apply(lambda x: x if 'X' not in x else None)
df = df.dropna(subset=['Sequence'])
print(len(df))
df['Sequence'] = df['Sequence'].apply(lambda x: x if 'B' not in x else None)
df = df.dropna(subset=['Sequence'])
print(len(df))
df['Sequence'] = df['Sequence'].apply(lambda x: x if 'Z' not in x else None)
df = df.dropna(subset=['Sequence'])
print(len(df))
df['Sequence'] = df['Sequence'].apply(lambda x: x if 'J' not in x else None)
df = df.dropna(subset=['Sequence'])
print(len(df))

# Curate the protein Sequences: drop sequences with repeated amino acids (more than 70% of the sequence)
df['Sequence'] = df['Sequence'].apply(lambda x: x if Counter(x).most_common(1)[0][1]/len(x) < 0.7 else None)
df = df.dropna(subset=['Sequence'])
print(len(df))

# Drop duplicates (Sequence and SMILES)
df = df.drop_duplicates(subset=['Sequence', 'SMILES'], keep='first')
print(len(df))

# Save the cleaned data
df.to_csv('../data/data_ChEMBL.csv')

# Save only the SMILES and Protein sequence of this cleaned data
df_simple = df[['Sequence', 'SMILES']]
df_simple.to_csv('../data/data_seqmol_ChEMBL_clean.csv')