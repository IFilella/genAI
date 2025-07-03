import regex as re
import pandas as pd
from curate_dataset_mols_prots import sanitize_molecules
from collections import Counter


regex = '(Br?|Cl?|Al|Ba|Ar|As|Au|Ba|Be|Bi|Ca|Cd|Ce|Co|Cr|Cu|Fe|Ga|Gd|Hg|La|Li|Mg|Mn|Mo|Na|Nb|Ni|Pd|Pt|Rb|Re|Ru|Sb|Sc|Se|Si|Sn|Sr|Ta|Tb|Tc|Te|Ti|U|V|W|Y|Yb|Zn|se|te|B|N|O|S|P|F|I|K|b|c|n|o|s|p|\\(|\\)|\\[|\\]|\\.|=|\n#|-|\\+|\\\\|\\/|:|~|@@|@|\\?|>>?|\\*|\\$|\\%|[0-9])'
regex = '(\\[[^\\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\.|=|\n#|-|\\+|\\\\|\\/|:|~|@|\\?|>>?|\\*|\\$|\\%[0-9]{2}|[0-9])'
regex = '(Br?|Cl?|As|Sb|Se|Si|Sn|Te|H|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\[|\\]|\\.|\\=|\\#|\\-|\\+|\\\\|\\/|\\:|\\~|@@|@|\\?|>>?|\\*|\\$|\\%[0-9]{2}|[0-9])'
regex = re.compile(regex)

def create_vocab(smiles):
    vocab = set()
    for smile in smiles:
        # smile = sanitize_molecules(smile)
        # if smile is None:
        #     continue
        tokens = regex.findall(smile)
        for token in tokens:
            vocab.add(token)
    return sorted(vocab)

def count_tokens_dataset(smiles):
    token_counts = {}
    for smile in smiles:
        tokens = regex.findall(smile)
        for token in tokens:
            if token not in token_counts:
                token_counts[token] = 0
            token_counts[token] += 1
    return token_counts

valid_atoms = {
    # Standard atoms
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Fl', 'Lv', 'Ts', 'Og',
    # Aromatic forms
    'b', 'c', 'n', 'o', 'p', 's'
}

def extract_atom(token):
    match = re.match(r'\[([A-Z][a-z]?|[a-z])', token)
    return match.group(1) if match else None

def count_atoms_dataset(smiles):
    tokenizer = re.compile(
    r'(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|B|C|b|c|n|o|s|p|\(|\)|\.|\=|\#|\-|\+|\\\\|\/|\:|\~|\@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])'
)
    
    atom_counts = Counter()

    for smi in smiles:
        tokens = tokenizer.findall(smi)
        for token in tokens:
            if token.startswith('['):
                atom = extract_atom(token)
            else:
                atom = token
            if atom in valid_atoms:
                atom_counts[atom] += 1

    return atom_counts

if __name__ == '__main__':
    df = pd.read_csv('../data/data_SMPBind_clean.csv')
    smiles = df['SMILES'].tolist()
    smiles = set(smiles)
            
    # Count the tokens in the dataset
    # token_counts = count_tokens_dataset(smiles)
    # token_counts_df = pd.DataFrame(list(token_counts.items()), columns=['token', 'count'])
    # token_counts_df = token_counts_df.sort_values(by='count', ascending=False)
    # print(token_counts_df)
    # token_counts_df.to_csv('token_counts_new.csv', index=False)
    
    # Count the atoms in the dataset
    # atom_counts = count_atoms_dataset(smiles)
    # atom_counts_df = pd.DataFrame(list(atom_counts.items()), columns=['atom', 'count'])
    # atom_counts_df = atom_counts_df.sort_values(by='count', ascending=False)
    # print(atom_counts_df)
    # atom_counts_df.to_csv('atom_counts.csv', index=False)
    
    # Create vocab from the dataset according to the regex
    vocab = create_vocab(smiles)
    print('Vocabulary size:', len(vocab))
    print('Vocabulary:', vocab)
    
    # Save the vocabulary to a file
    with open('../data/vocab.txt', 'w') as f:
        for token in vocab:
            f.write(token + '\n')