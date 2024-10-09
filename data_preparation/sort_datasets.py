import pandas as pd

df = pd.read_csv('../data/data_ChEMBL_BindingDB.csv', index_col=0)
def sort_df_by_len(df, col_mols, col_prots):
    df['mol_len'] = df[col_mols].apply(lambda x: len(str(x)))
    df['prot_len'] = df[col_prots].apply(lambda x: len(str(x)))
    df.mol_len = df.mol_len.astype(int)
    df.prot_len = df.prot_len.astype(int)
    df = df.sort_values(['mol_len', 'prot_len']).reset_index(drop=True).drop('mol_len', axis=1).drop('prot_len', axis=1)
    return df

df_sort = sort_df_by_len(df, 'SMILES', 'Sequence')
print(df_sort)

df_sort = df_sort[df_sort['SMILES'].notna()]
df_sort = df_sort[df_sort['SMILES'].map(lambda x: len(str(x)) > 2)]
print(df_sort)

# uppercase all protein sequences
df_sort['Sequence'] = df_sort['Sequence'].str.upper()
print(df_sort)

df_sort.to_csv('../data/data_ChEMBL_BindingDB_sort.csv')