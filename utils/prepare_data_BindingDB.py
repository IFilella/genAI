from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import multiprocessing as mp
import os


def sdf_to_csv(sdf_file, output_file, delimiter=','):
    """
    Convert an SDF file to a CSV or TSV file.
    
    Args:
        sdf_file (str): Path to the input SDF file.
        output_file (str): Path to the output CSV or TSV file.
        delimiter (str, optional): Delimiter to use in the output file. Default is ',' (CSV).
        
        THIS FUNCTION DOES NOT WORK WITH LARGE SDF FILES FOR MEMORY PROBLEMS!!
        I WILL USE THE FUNCTIONS BELOW TO CONVERT THE SDF FILE TO CSV
    """
    # Read the SDF file
    suppl = Chem.SDMolSupplier(sdf_file)

    # Convert to a pandas DataFrame
    df = PandasTools.LoadSDF(sdf_file, smilesName='SMILES', molColName='Molecule', includeFingerprints=False)

    # Save the DataFrame as CSV or TSV
    df.to_csv(output_file, sep=delimiter, index=False)
    
    print(f"Conversion complete! Output file: {output_file}")


sdf_file = 'data/BindingDB_All_2D_202406.sdf'
csv_file = 'data/BindingDB_All_2D_202406_prep.csv'
#sdf_to_csv(sdf_file, csv_file)

def process_molecule(mol):
    if mol is None:
        return None
    mol_dict = {}
    try:
        mol_dict['ID'] = mol.GetProp('_Name')
    except:
        mol_dict['ID'] = ''
    mol_dict['SMILES'] = Chem.MolToSmiles(mol)
    for prop in mol.GetPropNames():
        mol_dict[prop] = mol.GetProp(prop)
    return mol_dict

def worker(input_queue, output_queue):
    while True:
        chunk = input_queue.get()
        if chunk is None:
            break
        results = []
        supplier = Chem.SDMolSupplier()
        supplier.SetData(chunk)
        for mol in supplier:
            if mol:
                result = process_molecule(mol)
                if result:
                    results.append(result)
        output_queue.put(results)

def parallel_transform(sdf_file, csv_file, num_workers=4, chunk_size=100):
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    # Start worker processes
    processes = []
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(input_queue, output_queue))
        p.start()
        processes.append(p)

    print(f'Started {num_workers} worker processes.')

    # Read the SDF file and distribute chunks to workers
    with open(sdf_file, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if line.startswith("$$$$"):  # End of a molecule record in SDF
                if len(chunk) >= chunk_size:
                    input_queue.put("".join(chunk))
                    chunk = []
        if chunk:
            input_queue.put("".join(chunk))

    # Add sentinel values to stop the workers
    for _ in range(num_workers):
        input_queue.put(None)

    # Collect results from output queue
    results = []
    while any(p.is_alive() for p in processes) or not output_queue.empty():
        while not output_queue.empty():
            result = output_queue.get()
            if result:
                results.extend(result)

    # Wait for all workers to finish
    for p in processes:
        p.join()

    print(f'Collected {len(results)} results from output queue.')

    # Convert results to DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    print(f'Saved results to {csv_file}.')
    
    
if __name__ == '__main__':
    
    # Transform sdf to csv
    # sdf_file = 'data/raw/BindingDB_All_2D_202406.sdf'
    # csv_file = 'data/raw/BindingDB_All_2D_202406.csv'
    # num_workers = os.cpu_count()  # Use all available CPUs
    # parallel_transform(sdf_file, csv_file, num_workers=num_workers)
    
    # Create pickle file for faster loading
    # df = pd.read_csv(csv_file)
    # df.to_pickle(csv_file.replace('.csv', '.pkl'))

    # Clean the data
    df = pd.read_pickle('data/raw/BindingDB_All_2D_202406.pkl')
    print(df.columns)
    print(len(df))
    
    # Get columns with Homo Sapiens as a source organism --> NO
    # df = df[df['Target Source Organism According to Curator or DataSource'] == 'Homo sapiens']
    # print(len(df))
    
    # Drop rows with empty SMILES or target sequence
    df = df[df['SMILES'].notna()]
    df = df[df['BindingDB Target Chain Sequence'].notna()]
    print(len(df))
    
    # Drop rows with any type of affinity information
    df = df[df['Ki (nM)'].notna() | df['IC50 (nM)'].notna() | df['Kd (nM)'].notna() | df['EC50 (nM)'].notna()]
    print(len(df))
    
    # Save the filtered df into a new csv file
    df.to_csv('data/data_BindingDB.csv')
    
    # Save only the SMILES and Protein sequence of this cleaned data
    df_simple = df[['BindingDB Target Chain Sequence', 'SMILES']]
    df_simple = df_simple.rename(columns={'BindingDB Target Chain Sequence': 'Sequence'})
    df_simple.to_csv('data/data_seqmol_BindingDB.csv')
    
    # Filter only rows with proteins no longer than 540 amino acids and larger than 60 amino acids
    df_filt = df[df['BindingDB Target Chain Sequence'].map(lambda x: len(str(x)) < 540)]
    df_filt = df_filt[df_filt['BindingDB Target Chain Sequence'].map(lambda x: len(str(x)) > 60)]
    print(len(df_filt))
    
    # Save only the SMILES and Protein sequence of this cleaned data
    df_simple = df_filt[['BindingDB Target Chain Sequence', 'SMILES']]
    df_simple = df_simple.rename(columns={'BindingDB Target Chain Sequence': 'Sequence'})
    df_simple.to_csv('data/data_seqmol_BindingDB_filt.csv')
    
    # Finally, fix the protein sequences that are written like chunks
    df_simple['Sequence'] = df_simple['Sequence'].apply(lambda x: x.replace(' ','').replace('\n',''))
    df_simple.to_csv('data/data_seqmol_BindingDB_filt.csv')
    
    # Filter the molecules with SMILES length smaller than 80
    df_filt = df_filt[df_filt['SMILES'].map(lambda x: len(str(x)) < 80)]
    print(len(df_filt))
    df_simple.to_csv('data/data_seqmol_BindingDB_filt.csv')
    
    