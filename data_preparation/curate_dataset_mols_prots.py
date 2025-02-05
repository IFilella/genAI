import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize.rdMolStandardize import Cleanup
from chembl_structure_pipeline import standardizer
from multiprocessing import Pool, cpu_count

# Sanitize molecules
def sanitize_molecules(smile):
    remover = SaltRemover()
    try:
        # Check if the molecule is valid
        mol = Chem.MolFromSmiles(smile)
        # Filter dataset by heavy atoms
        atoms = mol.GetNumHeavyAtoms()
        if atoms < 4 or atoms > 70:
            return None
        else:
            # Remove salts and clean up the molecule
            a = remover.StripMol(mol, dontRemoveEverything=True)
            e = Cleanup(a)
            s = standardizer.standardize_mol(e)
            clean_smile = Chem.MolToSmiles(s)
            return clean_smile
    except:
        print(f"Failed to sanitize molecule: {smile}")
        return None

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('../data/data_ChEMBL_BindingDB_sort.csv', index_col=0)
    #df = pd.read_csv('../data/sample_1000.csv', index_col=0)
    print(df.shape)

    smiles = df['SMILES'].tolist()
    
    # Parallelize the sanitization process
    with Pool(cpu_count()) as p:
        clean_smiles = p.map(sanitize_molecules, smiles)
        
    # Filter out None values (where sanitization failed)
    df['SMILES'] = clean_smiles
    df = df.dropna(subset=['SMILES'])
    
    print(df)
    df.to_csv('../data/data_ChEMBL_BindingDB_clean.csv')