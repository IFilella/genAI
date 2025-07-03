import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1st run mmseqs to cluster the sequences
# mmseqs easy-cluster ../../data/data_SMPBind_seqs.fasta clusterRes tmp --min-seq-id 0.9

# Analysis of the clustering results
"""
df_clu = pd.read_csv('cluter_mmseqs_SMPBind/cluster95_cluster.tsv', sep='\t', header=None, names=['clu_rep', 'clu_member'])

# Create a dictionary of clusters and counts
clusters = df_clu.groupby('clu_rep')['clu_member'].apply(list).to_dict()
counts = df_clu['clu_rep'].value_counts()
counts_df = pd.DataFrame(counts).reset_index()

print('Number of clusters:', len(clusters))
print('Number of elements per cluster:\n', counts_df)

# change to just protein IDs
df_clu['clu_rep'] = df_clu['clu_rep'].astype(str)
df_clu['clu_member'] = df_clu['clu_member'].astype(str)
df_clu['clu_rep'] = df_clu['clu_rep'].str.split('|').str[0]
df_clu['clu_member'] = df_clu['clu_member'].str.split('|').str[0]

# Change the sequences from the original dataset to the representative of the cluster
df = pd.read_csv('../data/data_SMPBind_clean.csv')

mem_to_rep_id = df_clu.set_index('clu_member')['clu_rep'].to_dict()
df['rep_id'] = df['Uniprot'].map(mem_to_rep_id)

rep_id_to_seq = df.set_index('Uniprot')['Sequence'].to_dict()
df['rep_seq'] = df['rep_id'].map(rep_id_to_seq)

df = df[['Uniprot', 'Compound_ID', 'SMILES', 'Sequence', 'rep_id', 'rep_seq']]
df.to_csv('../data/data_SMPBind_clean_mapping.csv', index=False)
print(df)
"""
# Load the dataset
"""
df = pd.read_csv('../data/data_SMPBind_clean_mapping.csv')

# 2nd downsample the molecules from the clusters
def downsample_clusters(df, n_samples):
    # Group by representative ID and sample n_samples from each group
    downsampled_df = df.groupby('rep_id').apply(lambda x: x.sample(n=min(len(x), n_samples), random_state=42)).reset_index(drop=True)
    return downsampled_df

# Downsample to 100 samples per cluster
n_samples = 25
downsampled_df = downsample_clusters(df, n_samples)
print(downsampled_df.columns)
downsampled_df = downsampled_df[['Uniprot', 'Compound_ID', 'SMILES', 'rep_seq']]
downsampled_df.rename(columns={'rep_seq': 'Sequence'}, inplace=True)
print(downsampled_df)
downsampled_df.to_csv('../data/data_SMPBind_downsampled_25.csv', index=False)
"""

# Plot the distribution of the number of molecules per protein according to the different downsampling strategies
"""
df100 = pd.read_csv('../data/data_SMPBind_downsampled_100.csv')
df75 = pd.read_csv('../data/data_SMPBind_downsampled_75.csv')
df50 = pd.read_csv('../data/data_SMPBind_downsampled_50.csv')
df40 = pd.read_csv('../data/data_SMPBind_downsampled_40.csv')
df25 = pd.read_csv('../data/data_SMPBind_downsampled_25.csv')

dfs = [df100, df75, df50, df40, df25]
labels = ['100', '75', '50', '40', '25']
colors = sns.color_palette("Set2", len(dfs))

# Plot setup
plt.figure(figsize=(15, 5))

for df, label, color in zip(dfs, labels, colors):
    molecules_per_protein = df.groupby('Sequence')['SMILES'].nunique().reset_index()
    molecules_per_protein.columns = ['Sequence', 'Num_molecules']
    
    counts = molecules_per_protein['Num_molecules'].value_counts().sort_index()
    counts = counts.reset_index()
    counts.columns = ['Num_molecules', 'Count']
    counts['Count'] = counts['Count'] * counts['Num_molecules']
    counts = counts.sort_values(by='Num_molecules')
    
    # Plot with label and color
    plt.plot(counts['Num_molecules'], counts['Count'], label=f'Downsample {label} samples', marker='o', color=color)

# Final plot styling
plt.xlabel('Number of Molecules per Protein')
plt.ylabel('Protein Count')
plt.title('Distribution of Molecules per Protein (Downsampling Strategies)')
plt.legend(title='Strategy')
plt.tight_layout()
plt.grid(True)
plt.savefig('../data/plots/downsampling_effect.png')
"""

# Calculate the percentage of proteins that have the 50% of protein-molecule pairs
def calculate_percentage(df, threshold=0.5):
    mol_count = df.groupby('Sequence')['SMILES'].nunique().reset_index()
    mol_count.columns = ['Sequence', 'Num_molecules']
    mol_count = mol_count.sort_values(by='Num_molecules', ascending=False)
    
    mol_count['Cumulative_Count'] = mol_count['Num_molecules'].cumsum()
    total_count = mol_count['Num_molecules'].sum()
    
    half_total = total_count * threshold
    num_proteins_needed = mol_count[mol_count['Cumulative_Count'] <= half_total].shape[0] + 1
    
    percentage = (num_proteins_needed / len(mol_count)) * 100
    
    return percentage

df = pd.read_csv('../data/data_SMPBind_downsampled_25.csv')
percentage = calculate_percentage(df)
print(f'Percentage of proteins needed to cover 50% of protein-molecule pairs: {percentage:.2f}%')
    