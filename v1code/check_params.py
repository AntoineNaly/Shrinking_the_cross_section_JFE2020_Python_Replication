# -*- coding: utf-8 -*-

import copy

def check_params(s, defaults, required=None):
    """
    Verifies parameter structure and sets defaults for optional parameters.
    
    Parameters:
    - s: The parameters structure to check (dict)
    - defaults: The default values for optional parameters (dict)
    - required: The names of the required parameters (list of strings)
    
    Returns:
    - s: The parameters structure with the missing values replaced by defaults
    
    Note: This function is called by parse_config.py
    """
    if required is None:
        required = []
    
    # Check required fields
    for req in required:
        if req not in s:
            raise ValueError(f"Field '{req}' is required")
    
    # Deep copy defaults to avoid mutation
    result = copy.deepcopy(defaults)
    
    # Override defaults with provided values
    for key, value in s.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively merge nested dictionaries
            result[key] = check_params(value, result[key])
        else:
            result[key] = value
    
    return result