import yaml

def load_config(config_path):
    """Loads the configuration file and returns a dictionary of the parameters."""
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    if not isinstance(config['verbose'], int):
        raise ValueError('The verbose parameter must be set to 0, 1, or 2.')
    elif config['verbose'] not in [0, 1, 2]:
        raise ValueError('The verbose parameter must be set to 0, 1, or 2.')
        
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
        'weight_decay': config['weight_decay'],
        'betas': config['betas'],
        'weights_path': config['weights_path'],
        'validation_split': config['validation_split'],
        'get_wandb': config['get_wandb'],
        'num_gpus': config['num_gpus'],
        'verbose': config['verbose'],
        'wandb_project': config['wandb_project'],
        'wandb_name': config['wandb_name'],
        'prot_max_length': config['prot_max_length'],
        'mol_max_length': config['mol_max_length'],
        'es_patience': config['es_patience'],
        'es_delta': config['es_delta'],
    }
    
    # Add configuration for wandb if enabled
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
            "weight_decay": config['weight_decay'],
            "architecture": "Decoder-only",
            "dataset": "ChEMBL_BindingDB_sorted",
        },
        'wandb_name': config['wandb_name'] + '_dm' + str(config['d_model'])\
            + '_nh' + str(config['num_heads']) + '_ff' + str(config['ff_hidden_layer'])\
            + '_nl' + str(config['num_layers']) + '_lr' + str(config['learning_rate'])\
            + '_bt' + str(config['batch_size']) + '_dp' + str(config['dropout'])
    }
    
    return config_dict
