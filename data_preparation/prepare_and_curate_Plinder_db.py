from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import multiprocessing as mp
import os

"""
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

# Remove duplicate rows (same SMILES and sequence)
df = df.drop_duplicates(subset=['Sequence', 'SMILES'], keep='first')
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

# Save this filtered data
df_filt.to_csv('../data/data_plinder_filt.csv')

# Sort the dataframe by Sequence length and SMILES length
def sort_df_by_len(df, col_mols, col_prots):
    df['mol_len'] = df[col_mols].apply(lambda x: len(str(x)))
    df['prot_len'] = df[col_prots].apply(lambda x: len(str(x)))
    df.mol_len = df.mol_len.astype(int)
    df.prot_len = df.prot_len.astype(int)
    df = df.sort_values(['mol_len', 'prot_len']).reset_index(drop=True).drop('mol_len', axis=1).drop('prot_len', axis=1)
    return df
df_sort = sort_df_by_len(df_filt, 'SMILES', 'Sequence')
print(df_sort)

# Clean the SMILES
from curate_dataset_mols_prots import sanitize_molecules
df_sort['SMILES'] = df_sort['SMILES'].apply(sanitize_molecules)
df_sort = df_sort.dropna(subset=['SMILES'])
print(df_sort)
    
# Save the sorted and cleaned data
df_sort.to_csv('../data/data_plinder_filt_clean.csv')

# Join this df with the one from BindingDB and ChEMBL
df_chembl_bindingdb = pd.read_csv('../data/data_ChEMBL_BindingDB_clean.csv', index_col=0)
df_plinder = pd.read_csv('../data/data_plinder_filt_clean.csv', index_col=0)
print(len(df_chembl_bindingdb))
df_chembl_bindingdb = df_chembl_bindingdb.drop_duplicates(subset=['Sequence', 'SMILES'], keep='first')
print(len(df_chembl_bindingdb))
print(len(df_plinder))
df_plinder['source'] = 'Plinder'
df_plinder = df_plinder[['Sequence', 'SMILES', 'source']]

df_final = pd.concat([df_chembl_bindingdb, df_plinder], ignore_index=True)
print(len(df_final))

# Drop duplicates (same SMILES and sequence)
df_final = df_final.drop_duplicates(subset=['Sequence', 'SMILES'], keep='first')
print(len(df_final))
print(df_final)
df_final.to_csv('../data/data_ChEMBL_BindingDB_Plinder_clean.csv')
"""

# Sanitize protein sequences: remove sequences containing X, B, Z, J, and non-alphabetic characters
df_final = pd.read_csv('../data/data_ChEMBL_BindingDB_Plinder_clean.csv', index_col=0)
print(len(df_final))
df_final['Sequence'] = df_final['Sequence'].apply(lambda x: x if 'X' not in x else None)
df_final = df_final.dropna(subset=['Sequence'])
print(len(df_final))
df_final['Sequence'] = df_final['Sequence'].apply(lambda x: x if 'B' not in x else None)
df_final = df_final.dropna(subset=['Sequence'])
print(len(df_final))
df_final['Sequence'] = df_final['Sequence'].apply(lambda x: x if 'Z' not in x else None)
df_final = df_final.dropna(subset=['Sequence'])
print(len(df_final))
df_final['Sequence'] = df_final['Sequence'].apply(lambda x: x if 'J' not in x else None)
df_final = df_final.dropna(subset=['Sequence'])
print(len(df_final))

# Sanitize protein sequences: drop sequences with repeated amino acids (more than 90% of the sequence)
from collections import Counter
df_final['Sequence'] = df_final['Sequence'].apply(lambda x: x if Counter(x).most_common(1)[0][1]/len(x) <= 0.7 else None)
df_final = df_final.dropna(subset=['Sequence'])
print(len(df_final))
df_final.to_csv('../data/data_ChEMBL_BindingDB_Plinder_clean.csv', index=False)