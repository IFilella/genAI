from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import multiprocessing as mp
import os
from collections import Counter

# Open the data
df = pd.read_csv('../data/raw/plinder_db.csv', index_col=0)
print(df.columns)
df = df.rename(columns={'ligand_rdkit_canonical_smiles': 'SMILES', 'sequences': 'Sequence'})
print(df)
print(len(df))

# Drop rows with empty SMILES or target sequence
df = df[df['SMILES'].notna()]
df = df[df['Sequence'].notna()]
print(len(df))

# Filter only rows with proteins no longer than 540 amino acids and larger than 60 amino acids
df_filt = df[df['Sequence'].map(lambda x: len(str(x)) < 540)]
df_filt = df_filt[df_filt['Sequence'].map(lambda x: len(str(x)) > 60)]
print(len(df_filt))

# Filter the molecules with SMILES length smaller than 80 and larger than 4
df_filt = df_filt[df_filt['SMILES'].map(lambda x: len(str(x)) < 80)]
df_filt = df_filt[df_filt['SMILES'].map(lambda x: len(str(x)) > 4)]
print(len(df_filt))

# Just in case, fix the protein sequences that are written like chunks (there were some cases in BindingDB)
df_filt['Sequence'] = df_filt['Sequence'].apply(lambda x: x.replace(' ','').replace('\n',''))

# Just in case, uppercase all the protein sequences
df_filt['Sequence'] = df_filt['Sequence'].apply(lambda x: x.upper())

# Clean the SMILES: sanitize the molecules
from curate_dataset_mols_prots import sanitize_molecules
df_filt['SMILES'] = df_filt['SMILES'].apply(sanitize_molecules)
df_filt = df_filt.dropna(subset=['SMILES'])
print(df_filt)

# Curate the protein sequences: remove non-canonical amino acids
df_filt['Sequence'] = df_filt['Sequence'].apply(lambda x: x if 'X' not in x else None)
df_filt = df_filt.dropna(subset=['Sequence'])
print(len(df_filt))
df_filt['Sequence'] = df_filt['Sequence'].apply(lambda x: x if 'B' not in x else None)
df_filt = df_filt.dropna(subset=['Sequence'])
print(len(df_filt))
df_filt['Sequence'] = df_filt['Sequence'].apply(lambda x: x if 'Z' not in x else None)
df_filt = df_filt.dropna(subset=['Sequence'])
print(len(df_filt))
df_filt['Sequence'] = df_filt['Sequence'].apply(lambda x: x if 'J' not in x else None)
df_filt = df_filt.dropna(subset=['Sequence'])
print(len(df_filt))

# Curate the protein sequences: drop sequences with repeated amino acids (more than 70% of the sequence)
df_filt['Sequence'] = df_filt['Sequence'].apply(lambda x: x if Counter(x).most_common(1)[0][1]/len(x) <= 0.7 else None)
df_filt = df_filt.dropna(subset=['Sequence'])
print(len(df_filt))

# Drop duplicates (Sequence and SMILES)
df_filt = df_filt.drop_duplicates(subset=['Sequence', 'SMILES'], keep='first')
print(len(df_filt))

# Save the cleaned data
df_filt.to_csv('../data/data_plinder_clean.csv', index=False)

# Save only the SMILES and Protein sequence of this cleaned data
df_simple = df_filt[['Sequence', 'SMILES']]
df_simple.to_csv('../data/data_seqmol_Plinder_clean.csv', index=False)