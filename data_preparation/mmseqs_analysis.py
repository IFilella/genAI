import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

# Load the data
df = pd.read_csv('seq_sim_analysis/mmseqs/valDB_cluster.tsv', sep='\t', header=None, names=['clu_rep', 'clu_member'])
print(df)

# Create a dictionary of clusters and counts
clusters = df.groupby('clu_rep')['clu_member'].apply(list).to_dict()
counts = df['clu_rep'].value_counts()
counts_df = pd.DataFrame(counts).reset_index()

print('Number of clusters:', len(clusters))
print('Number of elements per cluster:\n', counts_df)

cluster_mapping = {rep: i for i, rep in enumerate(df['clu_rep'].unique())}

# Plot a tSNE of the clusters
matrix = pd.read_csv('seq_sim_analysis/validation_similarity_matrix.csv', index_col=0)
distance_matrix = 1 - matrix

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(distance_matrix)

tsne_df = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
tsne_df['Cluster'] = df['clu_rep'].map(cluster_mapping)

plt.figure(figsize=(10, 10))
sns.scatterplot(data=tsne_df, x='tsne1', y='tsne2', hue='Cluster', alpha=0.7, linewidth=0)
plt.title('tSNE of MMseqs Clusters - Validation Set')
plt.savefig('seq_sim_analysis/mmseqs/tSNE_validation.png')

# Take the 100 most populated clusters
top_clusters = counts_df['clu_rep'].values[:100]
print('Top clusters:', top_clusters)
print(len(top_clusters))

# Get the sequence of the representative of each cluster
df = pd.read_csv('seq_sim_analysis/tmp_val_dataset.csv', index_col=0)
top_cluster_seqs = df[df['Seq_ID'].isin(top_clusters)]
top_cluster_seqs = top_cluster_seqs.drop_duplicates(subset='Seq_ID', keep='first')
print(top_cluster_seqs)

top_cluster_seqs.to_csv('../tests/test_generation/cluster_representatives_100.csv')