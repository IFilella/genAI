import pandas as pd
from statistics import mode, median, stdev
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import time

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
# Drop the row with mol len lower than 4
df['SMILES'] = df['SMILES'].apply(lambda x: x if len(x) > 4 else None)
df = df.dropna()
min_mol_length = df['mol_len'].min()
min_mol_length_idx = df['mol_len'].idxmin()
print('Min SMILES length:', min_mol_length)
print('Min SMILES length index:', min_mol_length_idx)
print('Min SMILES:', df.loc[min_mol_length_idx, 'SMILES'])
print(len(df))

# Save this cleaned dataset
df_clean = df[['Sequence', 'SMILES', 'source']]
df_clean.to_csv('../data/data_ChEMBL_BindingDB_Plinder_clean.csv')

# Plot the distribution of the protein sequence lengths
plt.figure()
sns.histplot(data=df, x='seq_len', bins=100)
plt.savefig('../data/plots/seq_length_distribution.png')

# Plot the distribution of the SMILES lengths
plt.figure()
sns.histplot(data=df, x='mol_len', bins=100)
plt.savefig('../data/plots/smiles_length_distribution.png')

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

