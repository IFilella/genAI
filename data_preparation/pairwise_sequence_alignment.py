from Bio import SeqIO
from Bio.Align import PairwiseAligner
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# First, transform the sequences into a fasta file
"""
df = pd.read_csv('../data/data_ChEMBL_BindingDB_Plinder_clean.csv', index_col=0)
print(len(df))

# retrieve the sequence IDs from ChEMBL
df_chembl = pd.read_csv('../data/data_ChEMBL.csv')
df_filtered_chembl = df[df['source'] == 'ChEMBL']
seq_to_id_chembl = df_chembl.set_index('Sequence')['Target_CHEMBL'].to_dict()
df_filtered_chembl.loc[:, 'Seq_ID'] = df['Sequence'].map(seq_to_id_chembl)
print(df_filtered_chembl)

# retrieve the sequence IDs from BindingDB
df_bindingdb = pd.read_csv('../data/data_BindingDB.csv', index_col=0)
df_bindingdb = df_bindingdb.rename(columns={'BindingDB Target Chain Sequence': 'Sequence'})
df_filtered_bindingdb = df[df['source'] == 'BindingDB']
seq_to_id_bindingdb = df_bindingdb.set_index('Sequence')['ID'].to_dict()
df_filtered_bindingdb.loc[:, 'Seq_ID'] = df['Sequence'].map(seq_to_id_bindingdb)
print(df_filtered_bindingdb)

# retrieve the sequence IDs from Plinder
df_plinder = pd.read_csv('../data/data_Plinder.csv')
df_filtered_plinder = df[df['source'] == 'Plinder']
seq_to_id_plinder = df_plinder.set_index('Sequence')['sequence_id'].to_dict()
df_filtered_plinder.loc[:, 'Seq_ID'] = df['Sequence'].map(seq_to_id_plinder)
print(df_filtered_plinder)

# Join the dataframes
df_all = pd.concat([df_filtered_chembl, df_filtered_bindingdb, df_filtered_plinder])
print(len(df_all))
df_all.to_csv('tmp_dataset.csv')

# Save the sequences in a fasta file
df_all = pd.read_csv('tmp_dataset.csv')
with open('dataset_sequences.fasta', 'w') as f:
    for i, row in df_all.iterrows():
        f.write(f'>{row["Seq_ID"]}\n{row["Sequence"]}\n')
"""

# Pairwise sequence alignment
"""
from multiprocessing import Pool

sequences = {str(record.seq): record.id for record in SeqIO.parse('dataset_sequences.fasta', 'fasta')}
print(len(sequences))

aligner = PairwiseAligner()
aligner.mode = 'global'

ids = list(sequences.values())
seqs = list(sequences.keys())
n = len(ids)

# matrix = np.zeros((n, n))
# for i in range(n):
#     for j in range(n):
#         if i <= j:
#             alignment = aligner.align(seqs[i], seqs[j])[0]
#             matches = sum(1 for a,b in zip(alignment[0], alignment[1]) if a == b)
#             identity = matches / max(len(seqs[i]), len(seqs[j]))
#             print(i, j, identity)
#             matrix[i, j] = identity
#             matrix[j, i] = identity
        
def compute_identity(pair):
    i, j = pair
    alignment = aligner.align(seqs[i], seqs[j])[0]
    matches = sum(1 for a,b in zip(alignment[0], alignment[1]) if a == b)
    identity = matches / max(len(seqs[i]), len(seqs[j]))
    return i, j, identity
    
pairs = [(i, j) for i in range(n) for j in range(i, n)]

num_workers = 10
with Pool(num_workers) as pool:
    results = pool.map(compute_identity, pairs)

matrix = np.zeros((n, n))
for i, j, identity in results:
    print(i, j, identity)
    matrix[i, j] = identity
    matrix[j, i] = identity

df_matrix = pd.DataFrame(matrix, index=ids, columns=ids)
df_matrix.to_csv('sequence_similarity_matrix.csv')
print(df_matrix)
"""


# Plot a clustermap of the similarity matrix
df_matrix = pd.read_csv('sequence_similarity_matrix.csv', index_col=0)
print(df_matrix)

#"""
plt.figure()
sns.clustermap(df_matrix, vmin=0, vmax=1, robust=True)
plt.savefig('sequence_similarity_matrix_clustermap.png')
#"""

# Cluster the sequences based on the similarity matrix (DBSCAN)
from sklearn.cluster import DBSCAN

distance_matrix = 1 - df_matrix
distance_array = distance_matrix.values

dbscan = DBSCAN(eps=0.7, min_samples=2, metric='precomputed')
clusters = dbscan.fit_predict(distance_array)

identity_df = pd.DataFrame(columns=['Seq_ID', 'Cluster'])
identity_df['Seq_ID'] = df_matrix.index
identity_df["Cluster"] = clusters
identity_df.to_csv('DBSCAN_clusters.csv')

print(identity_df)

print(identity_df['Cluster'].value_counts())

# Plot the clusters in a tSNE plot
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X = tsne.fit_transform(distance_matrix)
tsne_df = pd.DataFrame(X, columns=['tsne1', 'tsne2'])
tsne_df['Cluster'] = clusters

plt.figure(figsize=(10, 10))
sns.scatterplot(data=tsne_df, x='tsne1', y='tsne2', hue='Cluster', alpha=0.7, linewidth=0)
plt.title('DBSCAN Clustering of Protein Sequences')
plt.legend(title='Cluster')
plt.savefig('DBSCAN_clusters_tsne.png')
