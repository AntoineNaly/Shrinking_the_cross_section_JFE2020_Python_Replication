# -*- coding: utf-8 -*-

from check_params import check_params

def parse_config(cfg, defaults):
    """
    Parse and process typical options in configuration dictionary.
    
    Parameters:
    - cfg: Configuration to process (dict or None)
    - defaults: Default configuration (dict)
    
    Returns:
    - cfg: The parsed configuration with defaults applied
    """
    # Make sure cfg is dict even if empty
    if cfg is None or not cfg:
        cfg = {}
    
    # Install defaults for whatever hasn't been specified
    result = check_params(cfg, defaults)
    
    return result