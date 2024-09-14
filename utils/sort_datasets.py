import pandas as pd

df = pd.read_csv('../data/data_ChEMBL_BindingDB.csv', index_col=0)
def sort_df_by_len(df, col_name):
    df['len'] = df[col_name].apply(lambda x: len(str(x)))
    df.len = df.len.astype(int)
    df = df.sort_values('len').reset_index(drop=True).drop('len', axis=1)
    return df

df_sort = sort_df_by_len(df, 'SMILES')
print(df_sort)

df_sort = df_sort[df_sort['SMILES'].notna()]
df_sort = df_sort[df_sort['SMILES'].map(lambda x: len(str(x)) > 2)]
print(df_sort)

df_sort.to_csv('../data/data_ChEMBL_BindingDB_sort.csv')