import pandas as pd

def convert_seqs_csv_to_fasta(input_csv, output_fasta, col_id='Uniprot', col_seq='Sequence', col_smiles='Compound_ID'):
    """
    Convert sequences from a CSV file to FASTA format.
    
    Parameters:
    input_csv (str): Path to the input CSV file containing sequences.
    output_fasta (str): Path to the output FASTA file.
    """
    df = pd.read_csv(input_csv)

    with open(output_fasta, 'w') as fasta_file:
        for index, row in df.iterrows():
            sequence = row[col_seq]
            seq_id = f'{str(row[col_id])}|{str(row[col_smiles])}'
            print(seq_id)
            fasta_file.write(f'>{seq_id}\n{sequence}\n')
            
if __name__ == '__main__':
    input_csv = '../data/data_SMPBind_clean.csv'
    output_fasta = '../data/data_SMPBind_seqs.fasta'
    
    convert_seqs_csv_to_fasta(input_csv, output_fasta)
    print(f'Successfully converted {input_csv} to {output_fasta}')