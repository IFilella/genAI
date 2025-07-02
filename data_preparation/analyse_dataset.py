import pandas as pd
from statistics import mode, median, stdev
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import time
"""
df = pd.read_csv('../data/data_ChEMBL_BindingDB_Plinder.csv', index_col=0)

# Calculate the sequence lengths of the proteins and the SMILES
df['seq_len'] = df['Sequence'].apply(lambda x: len(x))
df['mol_len'] = df['SMILES'].apply(lambda x: len(x))
print(df)

# Check the maximum sequence length
max_seq_length = df['seq_len'].max()
max_seq_length_idx = df['seq_len'].idxmax()
print('Max sequence length:', max_seq_length)
print('Max sequence length index:', max_seq_length_idx)
max_mol_length = df['mol_len'].max()
max_mol_length_idx = df['mol_len'].idxmax()
print('Max SMILES length:', max_mol_length)
print('Max SMILES length index:', max_mol_length_idx)
print('Max SMILES:', df.loc[max_mol_length_idx, 'SMILES'])
print('Max SMILES row:', df.loc[max_mol_length_idx])
# Drop the row with mol len higher than 80
df['SMILES'] = df['SMILES'].apply(lambda x: x if len(x) < 80 else None)
df = df.dropna()
max_mol_length = df['mol_len'].max()
max_mol_length_idx = df['mol_len'].idxmax()
print('Max SMILES length:', max_mol_length)
print('Max SMILES length index:', max_mol_length_idx)
print('Max SMILES:', df.loc[max_mol_length_idx, 'SMILES'])
print(len(df))

# Check the minimum sequence length
min_seq_length = df['seq_len'].min()
min_seq_length_idx = df['seq_len'].idxmin()
print('Min sequence length:', min_seq_length)
print('Min sequence length index:', min_seq_length_idx)
min_mol_length = df['mol_len'].min()
min_mol_length_idx = df['mol_len'].idxmin()
print('Min SMILES length:', min_mol_length)
print('Min SMILES length index:', min_mol_length_idx)
print('Min SMILES:', df.loc[min_mol_length_idx, 'SMILES'])
print('Min SMILES row:', df.loc[min_mol_length_idx])
# Drop the row with mol len lower than 4
df['SMILES'] = df['SMILES'].apply(lambda x: x if len(x) > 4 else None)
df = df.dropna()
min_mol_length = df['mol_len'].min()
min_mol_length_idx = df['mol_len'].idxmin()
print('Min SMILES length:', min_mol_length)
print('Min SMILES length index:', min_mol_length_idx)
print('Min SMILES:', df.loc[min_mol_length_idx, 'SMILES'])
print(len(df))

# Check the number of heavy atoms in the SMILES
from rdkit import Chem
df['atoms'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x).GetNumHeavyAtoms())
min_num_atoms = df['atoms'].min()
min_num_atoms_idx = df['atoms'].idxmin()
print('Min number of heavy atoms:', min_num_atoms)
print('Min number of heavy atoms index:', min_num_atoms_idx)
print('Min number of heavy atoms SMILES:', df.loc[min_num_atoms_idx, 'SMILES'])
max_num_atoms = df['atoms'].max()
max_num_atoms_idx = df['atoms'].idxmax()
print('Max number of heavy atoms:', max_num_atoms)
print('Max number of heavy atoms index:', max_num_atoms_idx)
print('Max number of heavy atoms SMILES:', df.loc[max_num_atoms_idx, 'SMILES'])
# Drop the row with less or equal to 4 heavy atoms
df = df[df['atoms'] > 4]
min_num_atoms = df['atoms'].min()
min_num_atoms_idx = df['atoms'].idxmin()
print('Min number of heavy atoms:', min_num_atoms)
print('Min number of heavy atoms index:', min_num_atoms_idx)
print('Min number of heavy atoms SMILES:', df.loc[min_num_atoms_idx, 'SMILES'])
print(len(df))
# Drop the row with more than 70 heavy atoms
df = df[df['atoms'] < 70]
max_num_atoms = df['atoms'].max()
max_num_atoms_idx = df['atoms'].idxmax()
print('Max number of heavy atoms:', max_num_atoms)
print('Max number of heavy atoms index:', max_num_atoms_idx)
print('Max number of heavy atoms SMILES:', df.loc[max_num_atoms_idx, 'SMILES'])
print(len(df))

# Save this cleaned dataset
df_clean = df[['Sequence', 'SMILES', 'source']]
df_clean.to_csv('../data/data_ChEMBL_BindingDB_Plinder_clean.csv')
"""

df = pd.read_csv('../data/data_SMPBind_clean.csv', index_col=0)
df['seq_len'] = df['Sequence'].apply(lambda x: len(x))
df['mol_len'] = df['SMILES'].apply(lambda x: len(x))

# Plot the distribution of the protein sequence lengths
plt.figure()
sns.histplot(data=df, x='seq_len', bins=100)
plt.savefig('../data/plots/SMPBind_seq_length_distribution.png')

# Plot the distribution of the SMILES lengths
plt.figure()
sns.histplot(data=df, x='mol_len', bins=100)
plt.savefig('../data/plots/SMPBind_smiles_length_distribution.png')

# Calculate the mean, median, mode, and standard deviation of the protein sequence lengths
mean_seq_length = df['Sequence'].apply(lambda x: len(x)).mean()
median_seq_length = df['Sequence'].apply(lambda x: len(x)).median()
mode_seq_length = mode(df['Sequence'].apply(lambda x: len(x)))
stdev_seq_length = df['Sequence'].apply(lambda x: len(x)).std()
print('Mean sequence length:', mean_seq_length)
print('Median sequence length:', median_seq_length)
print('Mode sequence length:', mode_seq_length)
print('Standard deviation sequence length:', stdev_seq_length)

# Calculate the mean, median, mode, and standard deviation of the SMILES lengths
mean_smiles_length = df['SMILES'].apply(lambda x: len(x)).mean()
median_smiles_length = df['SMILES'].apply(lambda x: len(x)).median()
mode_smiles_length = mode(df['SMILES'].apply(lambda x: len(x)))
stdev_smiles_length = df['SMILES'].apply(lambda x: len(x)).std()
print('Mean SMILES length:', mean_smiles_length)
print('Median SMILES length:', median_smiles_length)
print('Mode SMILES length:', mode_smiles_length)
print('Standard deviation SMILES length:', stdev_smiles_length)
exit()

# Plot the distribution of the protein sequence lengths per source
plt.figure()
sns.histplot(data=df, x='seq_len', hue='source', element='step')
plt.savefig('../data/plots/seq_length_distribution_per_source.png')

# Plot the distribution of the SMILES lengths per source
plt.figure()
sns.histplot(data=df, x='mol_len', hue='source', element='step')
plt.savefig('../data/plots/smiles_length_distribution_per_source.png')

# Calculate the mean, median, mode, and standard deviation of the protein sequence lengths per source
mean_seq_length_per_source = df.groupby('source')['Sequence'].apply(lambda x: x.apply(lambda y: len(y)).mean())
median_seq_length_per_source = df.groupby('source')['Sequence'].apply(lambda x: x.apply(lambda y: len(y)).median())
mode_seq_length_per_source = df.groupby('source')['Sequence'].apply(lambda x: mode(x.apply(lambda y: len(y))))
stdev_seq_length_per_source = df.groupby('source')['Sequence'].apply(lambda x: x.apply(lambda y: len(y)).std())
print('Mean sequence length per source:', mean_seq_length_per_source)
print('Median sequence length per source:', median_seq_length_per_source)
print('Mode sequence length per source:', mode_seq_length_per_source)
print('Standard deviation sequence length per source:', stdev_seq_length_per_source)

# Calculate the mean, median, mode, and standard deviation of the SMILES lengths per source
mean_smiles_length_per_source = df.groupby('source')['SMILES'].apply(lambda x: x.apply(lambda y: len(y)).mean())
median_smiles_length_per_source = df.groupby('source')['SMILES'].apply(lambda x: x.apply(lambda y: len(y)).median())
mode_smiles_length_per_source = df.groupby('source')['SMILES'].apply(lambda x: mode(x.apply(lambda y: len(y))))
stdev_smiles_length_per_source = df.groupby('source')['SMILES'].apply(lambda x: x.apply(lambda y: len(y)).std())
print('Mean SMILES length per source:', mean_smiles_length_per_source)
print('Median SMILES length per source:', median_smiles_length_per_source)
print('Mode SMILES length per source:', mode_smiles_length_per_source)
print('Standard deviation SMILES length per source:', stdev_smiles_length_per_source)


#df = pd.read_csv('../data/data_ChEMBL_BindingDB_Plinder_clean.csv', index_col=0)
df = pd.read_csv('../data/data_SMPBind_clean.csv', index_col=0)

# Calculate the number of different proteins and molecules that are present in the dataset
unique_sequences = df['Sequence'].nunique()
unique_molecules = df['SMILES'].nunique()
print('Number of unique sequences:', unique_sequences)
print('Number of unique molecules:', unique_molecules)

# Count the number of molecules per protein (group by Sequence)
molecules_per_protein = df.groupby('Sequence')['SMILES'].nunique().reset_index()
molecules_per_protein.columns = ['Sequence', 'Num_molecules']
molecules_per_protein = molecules_per_protein.sort_values(by='Num_molecules', ascending=False)
molecules_per_protein.to_csv('molecules_per_protein.csv')
print('Number of molecules per protein:\n', molecules_per_protein)

# Calculate the mean, median, mode, and standard deviation of the number of molecules per protein
mean_molecules_per_protein = molecules_per_protein['Num_molecules'].mean()
median_molecules_per_protein = molecules_per_protein['Num_molecules'].median()
mode_molecules_per_protein = mode(molecules_per_protein['Num_molecules'])
stdev_molecules_per_protein = molecules_per_protein['Num_molecules'].std()
print('Mean number of molecules per protein:', mean_molecules_per_protein)
print('Median number of molecules per protein:', median_molecules_per_protein)
print('Mode number of molecules per protein:', mode_molecules_per_protein)
print('Standard deviation number of molecules per protein:', stdev_molecules_per_protein)

# Plot the distribution of num. of molecules per protein sequence
plt.figure()
sns.histplot(data=molecules_per_protein, bins=50, kde=True)
plt.savefig('../data/plots/SMPBind_mols_per_prot_distribution.png')