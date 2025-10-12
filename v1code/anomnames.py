# -*- coding: utf-8 -*-

def anomnames(anom):
    """
    Convert anomaly codes to display names.
    
    Parameters:
    - anom: List of anomaly codes
    
    Returns:
    - names: List of formatted names for display
    """
    names = []
    
    for i in range(len(anom)):
        a = anom[i]
        
        # Handle different prefixes
        if len(a) >= 3:
            prefix = a[:3]
            s = a[3:] if len(a) > 3 else ''
            
            if prefix == 'rme':
                n = 'market'
            elif prefix == 're_':
                n = s
            elif prefix == 'r2_':
                n = f"{s}$^2$"
            elif prefix == 'r3_':
                n = f"{s}$^3$"
            elif prefix == 'rX_':
                # Handle interaction terms
                xsep = '__'
                idx = s.find(xsep)
                if idx == -1:
                    xsep = '_'  # OLD convention
                    idx = s.find(xsep)
                
                if idx != -1:
                    n = f"{s[:idx]}$\\times${s[idx+len(xsep):]}"
                else:
                    n = s
            else:
                # Check 2-character prefix
                if len(prefix) >= 2 and prefix[:2] == 'r_':
                    n = a[2:]
                else:
                    n = a
        else:
            n = a
        
        # Replace underscores for LaTeX
        n = n.replace('_', '\\_')
        
        # Convert to lowercase
        names.append(n.lower())
    
    return names