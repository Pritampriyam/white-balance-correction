import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance if using defaults
    if 'defaults' in config:
        for default_file in config['defaults']:
            default_path = os.path.join(os.path.dirname(config_path), f"{default_file}.yaml")
            default_config = load_config(default_path)
            config = {**default_config, **config}
        del config['defaults']
    
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

class ConfigDict:
    """Dictionary that allows attribute-style access"""
    
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
