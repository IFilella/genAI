import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import time

df_chembl = pd.read_csv('../data/data_seqmol_ChEMBL_clean.csv', index_col=0)
df_chembl['source'] = 'ChEMBL'
df_bindingdb = pd.read_csv('../data/data_seqmol_BindingDB_clean.csv', index_col=0)
df_bindingdb['source'] = 'BindingDB'
df_plinder = pd.read_csv('../data/data_seqmol_Plinder_clean.csv')
df_plinder['source'] = 'Plinder'
print('Number of samples in ChEMBL:', len(df_chembl))
print('Number of samples in BindingDB:', len(df_bindingdb))
print('Number of samples in Plinder:', len(df_plinder))

# Merge the datasets
df = pd.concat([df_chembl, df_bindingdb, df_plinder], ignore_index=True)
print('Number of samples in the merged dataset:', len(df))
print(df)

# Remove duplicates
df = df.drop_duplicates(subset=['Sequence', 'SMILES'], keep='first')
print('Number of samples after removing duplicates:', len(df))

# Save the merged dataset
df.to_csv('../data/data_ChEMBL_BindingDB_Plinder.csv')
