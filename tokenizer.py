import torch
from collections import Counter
from itertools import chain
from transformers import AutoTokenizer
import json

class MolecularTokenizer:
    def __init__(self):
        self.bos_token = '<bos>'
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'
        self.special_tokens = [self.bos_token, self.pad_token, self.eos_token, self.unk_token]
        self.vocab = None
        self.token2id = None
        self.id2token = None
        self.bos_token_id = None
        self.pad_token_id = None
        self.eos_token_id = None
        self.unk_token_id = None
        self.vocab_size = None

    def build_vocab(self, molecule_list):

        total_chars = Counter(chain(*molecule_list))
        self.vocab = list(total_chars.keys()) + self.special_tokens
        self.vocab_size = len(self.vocab)
        
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for idx, token in enumerate(self.vocab)}
        
        self.bos_token_id = self.token2id[self.bos_token]
        self.pad_token_id = self.token2id[self.pad_token]
        self.eos_token_id = self.token2id[self.eos_token]
        self.unk_token_id = self.token2id[self.unk_token]

    def __call__(self, molecule_list, truncation=True, padding='max_length', max_length=100, return_tensors="pt"):

        # Build vocabulary based on the given molecule list
        self.build_vocab(molecule_list)

        all_input_ids = []
        all_attention_masks = []

        for text in molecule_list:
            tokens = [x for x in text]
            
            if truncation and len(tokens) > max_length - 2: # -2 to make space for bos and eos tokens
                tokens = tokens[:max_length - 2]
            
            tokens = [self.bos_token] + tokens + [self.eos_token]
            
            if padding == 'max_length' and len(tokens) < max_length:
                tokens += [self.pad_token] * (max_length - len(tokens))
            
            input_ids = [self.token2id.get(token, self.unk_token_id) for token in tokens]
            attention_mask = [1 if token != self.pad_token else 0 for token in tokens]

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        input_ids_tensor = torch.tensor(all_input_ids)
        attention_masks_tensor = torch.tensor(all_attention_masks)

        return {'input_ids': input_ids_tensor, 'attention_mask': attention_masks_tensor}
    
    # To decode the new generated text from the model, we need to save the vocabulary mappings
    def save_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.vocab, f)
    
    def load_vocab(self, path):
        with open(path, 'r') as f:
            self.vocab = json.load(f)
            
        self.vocab_size = len(self.vocab)
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for idx, token in enumerate(self.vocab)}
        self.bos_token_id = self.token2id[self.bos_token]
        self.pad_token_id = self.token2id[self.pad_token]
        self.eos_token_id = self.token2id[self.eos_token]
        self.unk_token_id = self.token2id[self.unk_token]
        
    def decode(self, token_ids, skip_special_tokens=False):
        
        tokens = [self.id2token.get(id, self.unk_token) for id in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        else:
            tokens = [token for token in tokens]
        return ''.join(tokens)

class Tokenizer:
    def __init__(self, prot_tokenizer_name='facebook/esm2_t33_650M_UR50D', mol_tokenizer_name='inhouse', delim='<DELIM>'):

        self.prot_tokenizer = AutoTokenizer.from_pretrained(prot_tokenizer_name)
        
        self.mol_tokenizer_name = mol_tokenizer_name
        if self.mol_tokenizer_name == 'inhouse':
            self.mol_tokenizer = MolecularTokenizer()
        else:
            self.mol_tokenizer = AutoTokenizer.from_pretrained(mol_tokenizer_name, trust_remote_code=True)
            
        self.delim = delim
        
        self.combined_vocab = None
        self.token2id = None
        self.id2token = None
        
        self.special_tokens = [self.prot_tokenizer.pad_token, self.prot_tokenizer.cls_token, self.prot_tokenizer.sep_token, self.prot_tokenizer.mask_token,\
                                self.mol_tokenizer.pad_token, self.mol_tokenizer.bos_token, self.mol_tokenizer.eos_token, self.mol_tokenizer.unk_token]
    
    def build_combined_vocab(self, prots, mols):
        # Build separate vocabularies for proteins and molecules
        self.prot_vocab = self.prot_tokenizer.get_vocab()

        if self.mol_tokenizer_name == 'inhouse':
            self.mol_tokenizer.build_vocab(mols)
            self.mol_vocab = self.mol_tokenizer.token2id
        else:
            self.mol_vocab = self.mol_tokenizer.get_vocab()

        # Combine them in a shared dictionary with unique keys
        self.combined_vocab = dict(self.prot_vocab.items())
        self.combined_vocab[self.delim] = max(self.prot_vocab.values()) + 1 # Add delimiter token
        current_index = max(self.prot_vocab.values()) + 2 # Add molecular tokens starting from the next index
        for token, idx in self.mol_vocab.items():
            if token not in self.combined_vocab:
                self.combined_vocab[token] = current_index
                current_index += 1
        
        self.vocab_size = len(self.combined_vocab)
        self.token2id = self.combined_vocab
        self.id2token = {v: k for k, v in self.combined_vocab.items()}

    def save_combined_vocab(self, path):
        with open(path, 'w') as f:
            json.dump(self.combined_vocab, f)
            
    def load_combined_vocab(self, path):
        with open(path, 'r') as f:
            self.combined_vocab = json.load(f)

        self.vocab_size = len(self.combined_vocab.keys())
        self.token2id = self.combined_vocab
        self.id2token = {v: k for k, v in self.combined_vocab.items()}
        
    def tokenize_texts(self, prots, mols):
        # Tokenize separately for protein, delimiter, and molecular sequences
        tokenized_prots = self.prot_tokenizer(prots, padding='max_length', truncation=True, max_length=540, return_tensors='pt')

        if self.mol_tokenizer_name == 'inhouse':
            tokenized_mols = self.mol_tokenizer(mols, padding='max_length', max_length=50, return_tensors='pt')
        else:
            tokenized_mols = self.mol_tokenizer(mols, padding='max_length', max_length=50, return_tensors='pt')

        tokenized_delim = (torch.tensor([self.token2id[self.delim]] * len(prots))).unsqueeze(1)

        # Adjust token IDs for molecular tokenizer based on the combined vocabulary
        prot_vocab_size = len(self.prot_tokenizer.get_vocab())
        tokenized_mols['input_ids'] = tokenized_mols['input_ids'] + prot_vocab_size + 1
        
        input_tensor = torch.cat((tokenized_prots['input_ids'], tokenized_delim, tokenized_mols['input_ids']), dim=1)

        return input_tensor, self.vocab_size
    
    def tokenize_texts_from_loaded_vocab(self, prots, mols):
        # This function tokenizes using a previously loaded combined vocab
        if self.token2id is None or self.id2token is None:
            raise ValueError('Combined vocabulary not loaded. Load the combined vocabulary before tokenizing.\
                            This way, you will avoid problems of different length when trying to distribute processes.')
        
        tokenized_prots = self.prot_tokenizer(prots, padding='max_length', truncation=True, max_length=450, return_tensors='pt')
        
        tokenized_mols = []
        max_len = 50
        for mol in mols:
            mol_ids = [self.token2id.get(char, self.token2id.get('<unk>')) for char in mol]
            mol_ids = mol_ids[:max_len-2] + [self.token2id.get('<pad>')] * (max_len-2 - len(mol_ids))
            mol_ids = [self.token2id.get('<bos>')] + mol_ids + [self.token2id.get('<eos>')]
            tokenized_mols.append(mol_ids)

        tokenized_mols = torch.tensor(tokenized_mols)
        
        tokenized_delim = (torch.tensor([self.token2id[self.delim]] * len(prots))).unsqueeze(1)

        # Combine all token IDs into a single tensor
        input_tensor = torch.cat((tokenized_prots['input_ids'], tokenized_delim, tokenized_mols), dim=1)

        return input_tensor, self.vocab_size
    
    def __call__(self, prots, mols, use_loaded_vocab=False):
        if use_loaded_vocab:
            return self.tokenize_texts_from_loaded_vocab(prots, mols)
        else:
            return self.tokenize_texts(prots, mols)
    
    def get_padding_token_ids(self):
        prot_pad_token_id = self.prot_tokenizer.pad_token_id
        mol_pad_token_id = self.mol_tokenizer.pad_token_id + len(self.prot_vocab)
        
        return prot_pad_token_id, mol_pad_token_id
    
    def decode(self, token_ids, skip_special_tokens=True):
        decoded_tokens = []
        for token_id in token_ids:
            token = self.id2token.get(token_id, self.mol_tokenizer.unk_token)
            if skip_special_tokens and token in self.special_tokens:
                continue
            decoded_tokens.append(token)
        return ''.join(decoded_tokens)
    

if __name__ == '__main__':
    
    # Some trials here
    from data.fake_data import texts
    prot_list = [x.split('$')[0] for x in texts]
    molecule_list = [x.split('$')[-1] for x in texts]
    # print(prot_list)
    # print(molecule_list)
    
    # Example usage of molecular tokenizer
    """
    molecular_tokenizer = MolecularTokenizer()
    encoded_molecule = molecular_tokenizer(molecule_list, truncation=True, padding='max_length', max_length=15, return_tensors="pt")
    print(encoded_molecule['input_ids'])
    print(encoded_molecule['input_ids'][0])
    print(molecular_tokenizer.vocab_size)
    print(molecular_tokenizer.id2token)
    print(molecular_tokenizer.decode([19,  0,  0,  1,  2,  3,  4,  3,  0,  5,  2,  0,  0,  2, 21]))
    """
    
    # Example usage of protein tokenizer
    """
    prot_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    encoded_prots = prot_tokenizer(prot_list, padding='max_length', truncation=True, max_length=20, return_tensors='pt')
    print(encoded_prots['input_ids'])
    print(prot_tokenizer.vocab_size)
    print(prot_tokenizer.get_vocab())
    """
    
    # Example usage of Tokenizer
    
    tokenizer = Tokenizer(prot_tokenizer_name='facebook/esm2_t33_650M_UR50D', mol_tokenizer_name='inhouse')
    tokenizer.build_combined_vocab(prot_list, molecule_list)
    tokenizer.save_combined_vocab('combined_vocab_test.json')
    
    """ # TO DO: fix the offset of molecular tokens when tokenize without a loaded vocab
    encoded_texts = tokenizer(prot_list, molecule_list)
    input_tensor, vocab_size = encoded_texts
    
    print(input_tensor[0])
    print(vocab_size)
    print(tokenizer.combined_vocab)
    prot_pad_token_id, mol_pad_token_id = tokenizer.get_padding_token_ids()
    print(prot_pad_token_id, mol_pad_token_id)
    tokenizer.save_combined_vocab('combined_vocab_test.json')
    """
    
    tokenizer = Tokenizer()
    tokenizer.load_combined_vocab('combined_vocab_test.json')
    encoded_texts, vocab_size = tokenizer(prot_list, molecule_list, use_loaded_vocab=True)
    print(encoded_texts[0])
    print(vocab_size)
    print(tokenizer.combined_vocab)

    test_to_decode = [ 0, 20, 11,  9, 19, 15,  4,  7,  7,  7,  6,  5,  6,  6,  7,  6, 15,  8,
                        5,  4, 11, 12, 16,  4, 12, 16, 17, 21, 18,  7, 13,  9, 19, 13, 14, 11,
                        12,  9, 13,  8, 19, 10, 15, 16,  7,  7, 12, 13,  6,  9, 11,  7, 14, 20,
                        7,  4,  7,  6, 17,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        33, 23, 23, 34, 35, 28, 36, 28, 23, 37, 35, 23, 23, 35, 23, 23, 35, 23,
                        37,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                        1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]
    
    decoded_text = tokenizer.decode(test_to_decode, skip_special_tokens=True)
    print(decoded_text)

