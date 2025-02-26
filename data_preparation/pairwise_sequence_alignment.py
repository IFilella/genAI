from Bio import SeqIO
from Bio.Align import PairwiseAligner
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

def transform_sequences_to_fasta(df, out_file):
    
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
    df_all.to_csv('seq_sim_analysis/tmp_dataset.csv')

    # Save the sequences in a fasta file
    df_all = pd.read_csv('seq_sim_analysis/tmp_dataset.csv')
    with open('%s'%out_file, 'w') as f:
        for i, row in df_all.iterrows():
            f.write(f'>{row["Seq_ID"]}\n{row["Sequence"]}\n')

def pairwise_sequence_alignment(fasta_file, num_cpus, out_file):

    sequences = {str(record.seq): record.id for record in SeqIO.parse(fasta_file, 'fasta')}
    print(len(sequences))

    aligner = PairwiseAligner()
    aligner.mode = 'global'

    ids = list(sequences.values())
    seqs = list(sequences.keys())
    n = len(ids)
            
    def compute_identity(pair):
        i, j = pair
        alignment = aligner.align(seqs[i], seqs[j])[0]
        matches = sum(1 for a,b in zip(alignment[0], alignment[1]) if a == b)
        identity = matches / max(len(seqs[i]), len(seqs[j]))
        return i, j, identity
        
    pairs = [(i, j) for i in range(n) for j in range(i, n)]

    num_workers = num_cpus
    with Pool(num_workers) as pool:
        results = pool.map(compute_identity, pairs)

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
    print(df_matrix)

    distance_matrix = 1 - df_matrix
    distance_array = distance_matrix.values

    dbscan = DBSCAN(eps, min_samples, metric='precomputed')
    clusters = dbscan.fit_predict(distance_array)

    identity_df = pd.DataFrame(columns=['Seq_ID', 'Cluster'])
    identity_df['Seq_ID'] = df_matrix.index
    identity_df["Cluster"] = clusters
    identity_df.to_csv(out_file)

    print(identity_df)
    print(identity_df['Cluster'].value_counts())
    
    return clusters

def plot_tSNE_clusters(seq_sim_matrix, clusters, out_file, perplexity=30, random_state=42):
    df_matrix = pd.read_csv(seq_sim_matrix, index_col=0)
    print(df_matrix)

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


if __name__ == '__main__':
    
    # Load dataset
    df = pd.read_csv('../data/data_ChEMBL_BindingDB_Plinder_clean.csv', index_col=0)
    print(len(df))
    
    # First, transform the sequences into a fasta file
    transform_sequences_to_fasta(df, 'seq_sim_analysis/dataset_sequences.fasta')
    
    # Pairwise sequence alignment of the fasta file
    pairwise_sequence_alignment('seq_sim_analysis/dataset_sequences.fasta', 10, 'seq_sim_analysis/sequence_similarity_matrix.csv')
    
    # Plot a clustermap of the similarity matrix
    clustermap_seq_sim_matrix('seq_sim_analysis/sequence_similarity_matrix.csv', '../data/plots/sequence_similarity_matrix_clustermap.png')
    
    # Cluster the sequences based on the similarity matrix (DBSCAN)
    clusters = cluster_simmatrix_DBSCAN('seq_sim_analysis/sequence_similarity_matrix.csv', 'seq_sim_analysis/DBSCAN_clusters.csv', 0.7, 2)
    
    # Plot the clusters in a tSNE plot
    plot_tSNE_clusters('seq_sim_analysis/sequence_similarity_matrix.csv', clusters, '../data/plots/DBSCAN_clusters_tsne.png')