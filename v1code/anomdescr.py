# -*- coding: utf-8 -*-

from characteristics_names_map import characteristics_names_map

def anomdescr(anom):
    """
    Get descriptions for anomaly names.
    
    Parameters:
    - anom: List of anomaly codes
    
    Returns:
    - descriptions: List of descriptions
    """
    if not isinstance(anom, list):
        raise ValueError("The first parameter must be a list")
    
    # Get the mapping
    desc_map = characteristics_names_map()
    
    descriptions = []
    
    for i in range(len(anom)):
        a = anom[i]
        
        # Handle different prefixes
        if len(a) >= 3:
            prefix = a[:3]
            s = a[3:] if len(a) > 3 else ''
            
            if prefix == 'rme':
                n = 'Market'
            elif prefix == 're_':
                n = read_desc(desc_map, s)
            elif prefix == 'r2_':
                n = f"{read_desc(desc_map, s)}$^2$"
            elif prefix == 'r3_':
                n = f"{read_desc(desc_map, s)}$^3$"
            elif prefix == 'rX_':
                # Handle interaction terms
                xsep = '__'
                idx = s.find(xsep)
                if idx == -1:
                    xsep = '_'  # OLD convention
                    idx = s.find(xsep)
                
                if idx != -1:
                    n = f"{read_desc(desc_map, s[:idx])}$\\times${read_desc(desc_map, s[idx+len(xsep):])}"
                else:
                    n = read_desc(desc_map, s)
            else:
                # Check 2-character prefix
                if len(prefix) >= 2 and prefix[:2] == 'r_':
                    s = a[2:]
                else:
                    s = a
                n = read_desc(desc_map, s)
        else:
            n = read_desc(desc_map, a)
        
        # Replace underscores for LaTeX
        n = n.replace('_', '\\_')
        
        descriptions.append(n)
    
    return descriptions


def read_desc(desc_map, s):
    """
    Read description from the map, but do not throw an error if no desc is available.
    
    Parameters:
    - desc_map: Dictionary mapping names to descriptions
    - s: Key to look up
    
    Returns:
    - desc: Description or original key if not found
    """
    if s in desc_map:
        return desc_map[s]
    else:
        # For FF25 portfolios, try to create a meaningful description
        if 'ME' in s and 'BM' in s:
            # Extract size and value indicators
            size_num = s[2] if len(s) > 2 and s[2].isdigit() else ''
            value_num = s[-1] if s[-1].isdigit() else ''
            return f"Size {size_num} Value {value_num}"
        elif 'SMALL' in s:
            if 'LoBM' in s:
                return 'Small Growth'
            elif 'HiBM' in s:
                return 'Small Value'
            else:
                return 'Small'
        elif 'BIG' in s:
            if 'LoBM' in s:
                return 'Large Growth'
            elif 'HiBM' in s:
                return 'Large Value'
            else:
                return 'Large'
        else:
            print(f"Warning: No description available for [{s}]")
            return s