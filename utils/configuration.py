import yaml

def load_config(config_path):
    """Loads the configuration file and returns a dictionary of the parameters."""
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # Load configuration variables
    config_dict = {
        'data_path': config['data_path'],
        'col_prots': config['col_prots'],
        'col_mols': config['col_mols'],
        'd_model': config['d_model'],
        'num_heads': config['num_heads'],
        'ff_hidden_layer': config['ff_hidden_layer'],
        'dropout': config['dropout'],
        'num_layers': config['num_layers'],
        'batch_size': config['batch_size'],
        'num_epochs': config['num_epochs'],
        'learning_rate': config['learning_rate'],
        'loss_function': config['loss_function'],
        'optimizer': config['optimizer'],
        'weights_path': config['weights_path'],
        'validation_split': config['validation_split'],
        'get_wandb': config['get_wandb'],
        'num_gpus': config['num_gpus'],
        'verbose': config['verbose'],
        'wandb_project': config['wandb_project'],
        'prot_max_length': config['prot_max_length'],
        'mol_max_length': config['mol_max_length'],
        'es_patience': config['es_patience'],
        'es_delta': config['es_delta'],
    }
    
    # Add configuration for wandb if enabled
    if config['get_wandb']:
        config_dict['wandb'] = {
            'wandb_project': config['wandb_project'],
            'wandb_config': {
                "learning_rate": config['learning_rate'],
                "batch_size": config['batch_size'],
                "num_epochs": config['num_epochs'],
                "d_model": config['d_model'],
                "num_heads": config['num_heads'],
                "ff_hidden_layer": config['ff_hidden_layer'],
                "dropout": config['dropout'],
                "num_layers": config['num_layers'],
                "architecture": "Decoder-only",
                "dataset": "ChEMBL_BindingDB_sorted_sample10000",
            }
        }
    
    return config_dict
