from plinder.core.scores import query_index
from plinder.core import PlinderSystem
import gc

cols_of_interest = ["system_id", "entry_pdb_id", "system_type", "ligand_plip_type",\
                    "ligand_rdkit_canonical_smiles", "ligand_is_covalent", "ligand_is_ion",\
                    "ligand_is_lipinski", "ligand_is_fragment", "ligand_is_oligo", "ligand_is_cofactor",\
                    "ligand_is_artifact", "ligand_is_other", "ligand_id"]
filters = [("system_num_ligand_chains", "==", 1), ("system_type", "==", "holo"),\
           ("ligand_plip_type", "==", "SMALLMOLECULE"), ("ligand_is_covalent", "==", False),\
           ("ligand_is_ion", "==", False), ("ligand_is_fragment", "==", False),\
           ("ligand_is_oligo", "==", False), ("ligand_is_cofactor", "==", False), \
           ("ligand_is_artifact", "==", False), ("ligand_is_other", "==", False)]
df = query_index(columns=cols_of_interest, filters=filters)
print(df)
df_clean = df[["system_id", "ligand_id", "ligand_rdkit_canonical_smiles"]]

chunk_size = 100
with open("plinder_db.csv", "w") as f:
    # Write header
    f.write("system_id,ligand_id,ligand_rdkit_canonical_smiles,sequence_id,sequences\n")
    # Write data in chunks
    for i in range(0, len(df_clean), chunk_size):
        print(i)
        chunk = df_clean.iloc[i:i + chunk_size]
        chunk["seq_id"] = chunk["system_id"].apply(lambda id: id.split("__")[2])
        def fetch_sequences(system_id):
            try:
                return PlinderSystem(system_id=system_id).sequences
            except Exception as e:
                print(f"Error fetching sequences for system_id {system_id}: {e}")
                return {}
        chunk["sequences"] = chunk["system_id"].apply(fetch_sequences)
        #chunk["sequences"] = chunk["system_id"].apply(lambda id: PlinderSystem(system_id=id).sequences)
        chunk["sequences"] = chunk["sequences"].apply(lambda seq_dict: next(iter(seq_dict.values()), None))
        chunk.to_csv(f, index=False, header=False, mode='a')
        del chunk
        gc.collect()
