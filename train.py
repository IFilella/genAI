import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from decoder import Model
import wandb

get_wandb = False

# TO DO: FER CONFIG FILE (YAML) containing these hyperparameters:

if get_wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="test-genAI",

        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "Decoder-only",
        "dataset": "Test dataset",
        "epochs": 10,
        }
    )

# Define parameters
d_model = 500
num_heads = 4
d_ff = 400
max_len = 200

# Get the input tensor

from data.fake_data import texts
# Separate proteins and SMILES strings
prot_seqs = [text.split('$')[0] for text in texts]
smiles = [text.split('$')[1] for text in texts]
delim = ['$'] * len(texts)

def tokenize_texts(prots, mols, prot_tokenizer='facebook/esm2_t33_650M_UR50D', mol_tokenizer='ibm/MolFormer-XL-both-10pct'):
    prot_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    mol_tokenizer = AutoTokenizer.from_pretrained(mol_tokenizer, trust_remote_code=True)
    
    tokenized_prots = prot_tokenizer(prots, padding='longest', return_tensors='pt')
    tokenized_mols = mol_tokenizer(mols, padding='longest', return_tensors='pt')
    tokenized_delim = prot_tokenizer(delim, padding=True, return_tensors='pt')
    
    input_tensor = torch.cat((tokenized_prots['input_ids'], tokenized_delim['input_ids'], tokenized_mols['input_ids']), dim=1)
    vocab_size = prot_tokenizer.vocab_size + mol_tokenizer.vocab_size + 1
    
    return input_tensor, vocab_size

input_tensor, vocab_size = tokenize_texts(prot_seqs, smiles, prot_tokenizer='facebook/esm2_t33_650M_UR50', mol_tokenizer='ibm/MolFormer-XL-both-10pct')
print(input_tensor.shape)

# Create dummy labels (at the moment, they are just random integers. No real data!!)
labels = torch.randint(0, vocab_size, (len(texts), input_tensor.shape[1]))

# Create a TensorDataset and DataLoader
dataset = TensorDataset(input_tensor, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Instantiate the model
model = Model(d_model, num_heads, d_ff, max_len, vocab_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 10
best_accuracy = 0

for epoch in range(num_epochs):
    model.to(device)
    
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        input_tensor, labels = batch
        input_tensor = input_tensor.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits, probs = model(input_tensor)

        # Reshape the logits and labels for loss calculation
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
            
        loss = criterion(logits, labels)
    
        loss.backward()
        optimizer.step()
    
        _, preds = torch.max(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.numel()

        total_loss += loss.item()
        
    acc = total_correct / total_samples
        
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {acc}")

    if get_wandb:
        # log metrics to wandb
        wandb.log({"Epoch": epoch+1, "loss": loss.item(), "accuracy": acc})
        
    # Save the model weights if accuracy improves
    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(model.state_dict(), 'weights/best_model_weights.pth')
    
print('Training complete!!!')
    
if get_wandb:    
    # [optional] finish the wandb run
    wandb.finish()

    

    
    