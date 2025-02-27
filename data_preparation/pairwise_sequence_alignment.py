from Bio import SeqIO
from Bio.Align import PairwiseAligner
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform
from collections import Counter

def transform_sequences_to_fasta(df_csv, out_file):
    df = pd.read_csv(df_csv)
    
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
    nan_indices = df_filtered_bindingdb['Seq_ID'].isna() | (df_filtered_bindingdb['Seq_ID'] == ' ')
    df_filtered_bindingdb.loc[nan_indices, 'Seq_ID'] = [f'BindingDB_{i}' for i in range(1, nan_indices.sum() + 1)]
    print(df_filtered_bindingdb)
    
    # retrieve the sequence IDs from Plinder
    df_plinder = pd.read_csv('../data/data_Plinder.csv')
    df_filtered_plinder = df[df['source'] == 'Plinder']
    # seq_to_id_plinder = df_plinder.set_index('Sequence')['sequence_id'].to_dict()
    # df_filtered_plinder.loc[:, 'Seq_ID'] = df['Sequence'].map(seq_to_id_plinder)
    # because these IDs are not unique, we will assign new IDs
    df_filtered_plinder.loc[:, 'Seq_ID'] = [f'Plinder_{i}' for i in range(1, len(df_filtered_plinder) + 1)]
    print(df_filtered_plinder)
    
    # Join the dataframes
    df_all = pd.concat([df_filtered_chembl, df_filtered_bindingdb, df_filtered_plinder])
    print(len(df_all))
    df_all.to_csv('seq_sim_analysis/tmp_dataset.csv')
    
    # Save the sequences in a fasta file
    df_all = pd.read_csv('seq_sim_analysis/tmp_dataset.csv')
    with open('%s'%out_file, 'w') as f:
        for i, row in df_all.iterrows():
            f.write(f'>{row["Seq_ID"]}\n{row["Sequence"]}\n')

def compute_identity(pair, seqs, aligner):
        i, j = pair
        alignment = aligner.align(seqs[i], seqs[j])[0]
        matches = sum(1 for a,b in zip(alignment[0], alignment[1]) if a == b)
        identity = matches / max(len(seqs[i]), len(seqs[j]))
        return i, j, identity
    
def pairwise_sequence_alignment(fasta_file, num_cpus, out_file):

    sequences = {str(record.seq): record.id for record in SeqIO.parse(fasta_file, 'fasta')}
    print(len(sequences))

    aligner = PairwiseAligner()
    aligner.mode = 'global'

    ids = list(sequences.values())
    seqs = list(sequences.keys())
    n = len(ids)

    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    
    from functools import partial
    compute_identity_partial = partial(compute_identity, seqs=seqs, aligner=aligner)
    
    num_workers = num_cpus
    with Pool(num_workers) as pool:
        results = pool.map(compute_identity_partial, pairs)

    matrix = np.zeros((n, n))
    for i, j, identity in results:
        print(i, j, identity)
        matrix[i, j] = identity
        matrix[j, i] = identity

    df_matrix = pd.DataFrame(matrix, index=ids, columns=ids)
    df_matrix.to_csv(out_file)
    print(df_matrix)

def clustermap_seq_sim_matrix(seq_sim_matrix, out_file):
    df_matrix = pd.read_csv(seq_sim_matrix, index_col=0)
    print(df_matrix)

    plt.figure()
    sns.clustermap(df_matrix, vmin=0, vmax=1, robust=True)
    plt.savefig(out_file)

def cluster_simmatrix_DBSCAN(seq_sim_matrix, out_file, eps, min_samples):
    df_matrix = pd.read_csv(seq_sim_matrix, index_col=0)

    distance_matrix = 1 - df_matrix
    distance_array = distance_matrix.values

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    clusters = dbscan.fit(distance_array).labels_
    
    counts = Counter(clusters)
    counts_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    print('Number of clusters:', len(counts))
    print('Cluster sizes:\n', counts_df)

    df_clusters = pd.DataFrame(clusters, index=df_matrix.index, columns=['Cluster'])
    df_clusters.to_csv(out_file)

    return clusters

def cluster_simmatrix_hierarchical(seq_sim_matrix, out_file, max_dist):
    df_matrix = pd.read_csv(seq_sim_matrix, index_col=0)

    distance_matrix = 1 - df_matrix
    distance_array = distance_matrix.values

    linkage_matrix = shc.linkage(squareform(distance_array), method='average')
    
    plt.figure(figsize=(50, 10))
    shc.dendrogram(linkage_matrix, labels=df_matrix.index, leaf_rotation=90)
    plt.axhline(y=max_dist, color='r', linestyle='--')
    plt.xlabel('Protein Sequence')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering of Protein Sequences')
    plt.savefig('../data/plots/hierarchical_clustering.png')
    
    threshold = max_dist
    clusters = shc.fcluster(linkage_matrix, threshold, criterion='distance')
    counts = Counter(clusters)
    counts_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    
    print('Number of clusters:', len(counts))
    print('Cluster sizes:\n', counts_df)
    
    df_clusters = pd.DataFrame(clusters, index=df_matrix.index, columns=['Cluster'])
    df_clusters.to_csv(out_file)
    
    return clusters

def plot_tSNE_clusters(seq_sim_matrix, clusters, out_file, perplexity=30, random_state=42):
    df_matrix = pd.read_csv(seq_sim_matrix, index_col=0)
    distance_matrix = 1 - df_matrix

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    X = tsne.fit_transform(distance_matrix)
    tsne_df = pd.DataFrame(X, columns=['tsne1', 'tsne2'])
    tsne_df['Cluster'] = clusters

    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=tsne_df, x='tsne1', y='tsne2', hue='Cluster', alpha=0.7, linewidth=0)
    plt.title('DBSCAN Clustering of Protein Sequences')
    plt.legend(title='Cluster')
    plt.savefig(out_file)

def map_seq_ids_to_fasta(df_ref, df_to_map, out_file):
    df_ref = pd.read_csv(df_ref, index_col=0)
    df_to_map = pd.read_csv(df_to_map, index_col=0)
    
    # Drop duplicates from validation split
    print(len(df_to_map))
    df_to_map = df_to_map.drop_duplicates(subset='Sequence', keep='first')
    print(len(df_to_map))
    
    seq_to_id = df_ref.set_index('Sequence')['Seq_ID'].to_dict()
    df_to_map.loc[:, 'Seq_ID'] = df_to_map['Sequence'].map(seq_to_id)
    
    df_to_map['Seq_ID'] = df_to_map['Seq_ID'].str.replace(' ', '')
    df_to_map['Sequence'] = df_to_map['Sequence'].str.replace(' ', '')
    nan = (df_to_map['Seq_ID'] == '')
    df_to_map.loc[nan, 'Seq_ID'] = [f'val_{i}' for i in range(1, nan.sum() + 1)]
    df_to_map.to_csv('seq_sim_analysis/tmp_val_dataset.csv')
    print(df_to_map)
    
    with open('%s'%out_file, 'w') as f:
        for i, row in df_to_map.iterrows():
            f.write(f'>{row["Seq_ID"]}\n{row["Sequence"]}\n')

if __name__ == '__main__':
    
    # ALL DATASET
    # First, transform the sequences into a fasta file
    #transform_sequences_to_fasta('../data/data_ChEMBL_BindingDB_Plinder_clean.csv', 'seq_sim_analysis/dataset_sequences.fasta')
    
    # Pairwise sequence alignment of the fasta file
    #pairwise_sequence_alignment('seq_sim_analysis/dataset_sequences.fasta', 10, 'seq_sim_analysis/sequence_similarity_matrix.csv')
    
    # Plot a clustermap of the similarity matrix
    #clustermap_seq_sim_matrix('seq_sim_analysis/sequence_similarity_matrix.csv', '../data/plots/sequence_similarity_matrix_clustermap.png')
    
    # Cluster the sequences based on the similarity matrix (DBSCAN)
    #clusters = cluster_simmatrix_DBSCAN('seq_sim_analysis/sequence_similarity_matrix.csv', 'seq_sim_analysis/DBSCAN_clusters.csv', 0.6, 2)
    
    # Cluster the sequences based on the similarity matrix (Hierarchical)
    #clusters = cluster_simmatrix_hierarchical('seq_sim_analysis/sequence_similarity_matrix.csv', 'seq_sim_analysis/hierarchical_clusters.csv', 0.6)
    
    # Plot the clusters in a tSNE plot
    #plot_tSNE_clusters('seq_sim_analysis/sequence_similarity_matrix.csv', clusters, '../data/plots/hierarchical_clusters_tsne.png')
    
    # JUST VALIDATION SPLIT    
    #map_seq_ids_to_fasta('seq_sim_analysis/tmp_dataset.csv', '../data/splits/validation_split.csv', 'seq_sim_analysis/validation_sequences.fasta')
    
    pairwise_sequence_alignment('seq_sim_analysis/validation_sequences.fasta', 10, 'seq_sim_analysis/validation_similarity_matrix.csv')
    