import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience # Number of epochs to wait before stopping training
        self.delta = delta # Minimum change in the monitored quantity to qualify as an improvement
        self.counter = 0 # Counter to keep track of the number of epochs with no improvement
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
        
    def __call__(self, val_loss, model, weights_path, fabric):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, weights_path, fabric)
            if self.verbose >= 1:
                fabric.print(f'EarlyStopping: Validation score improved ({self.best_score:.6f} --> {score:.6f}).')
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose >= 1:
                fabric.print(f'EarlyStopping: EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, weights_path, fabric)
            self.counter = 0
            if self.verbose >= 1:
                fabric.print(f'EarlyStopping: Validation score improved ({self.best_score:.6f} --> {score:.6f}).  Resetting counter to 0.')
            
    def save_checkpoint(self, model, weights_path, fabric):
        state = {'model': model}
        fabric.save(weights_path, state)
