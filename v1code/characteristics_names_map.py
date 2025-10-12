# -*- coding: utf-8 -*-

def characteristics_names_map():
    """
    Create a mapping from characteristic names to descriptions.
    
    Returns:
    - map: Dictionary mapping names to descriptions
    """
    # Anomalies
    anom_names = [
        'size', 'value', 'prof', 'valprof', 'fscore', 'debtiss', 'repurch', 
        'nissa', 'accruals', 'growth', 'aturnover', 'gmargins',
        'divp', 'ep', 'cfp', 'noa', 'inv', 'invcap', 'igrowth', 'sgrowth', 
        'lev', 'roaa', 'roea', 'sp', 'gltnoa',
        'mom', 'indmom', 'valmom', 'valmomprof', 'shortint', 'mom12', 
        'momrev', 'lrrev', 'valuem', 'nissm', 'sue', 'roe', 'rome', 'roa', 
        'strev', 'ivol', 'betaarb',
        'season', 'indrrev', 'indrrevlv', 'indmomrev', 'ciss', 'price', 
        'age', 'shvol'
    ]
    
    anom_descriptions = [
        'Size', 'Value (A)', 'Gross profitability', 'Value-profitability', 
        'F-score', 'Debt issuance', 'Share repurchases',
        'Net issuance (A)', 'Accruals', 'Asset growth', 'Asset turnover', 
        'Gross margins',
        'Dividend/Price', 'Earnings/Price', 'Cash Flows/Price', 
        'Net operating assets', 'Investment/Assets', 'Investment/Capital', 
        'Investment growth', 'Sales growth',
        'Leverage', 'Return on assets (A)', 'Return on book equity (A)', 
        'Sales/Price', 'Growth in LTNOA',
        'Momentum (6m)', 'Industry momentum', 'Value-momentum', 
        'Value-momentum-prof.', 'Short interest', 'Momentum (12m)',
        'Momentum-reversals', 'Long-run reversals', 'Value (M)', 
        'Net issuance (M)', 'Earnings surprises',
        'Return on book equity (Q)', 'Return on market equity', 
        'Return on assets (Q)', 'Short-term reversals',
        'Idiosyncratic volatility', 'Beta arbitrage', 'Seasonality', 
        'Industry rel. reversals',
        'Industry rel. rev. (L.V.)', 'Ind. mom-reversals', 'Composite issuance', 
        'Price', 'Age', 'Share volume'
    ]
    
    # Create anomalies mapping
    map_anomalies = dict(zip(anom_names, anom_descriptions))
    
    # FF25 portfolio names mapping
    ff25_names = []
    ff25_descriptions = []
    
    # Size quintiles
    size_labels = ['Small', 'ME2', 'ME3', 'ME4', 'Big']
    # Book-to-market quintiles
    bm_labels = ['Low', 'BM2', 'BM3', 'BM4', 'High']
    
    # Generate all combinations
    for i, size in enumerate(['SMALL', 'ME2', 'ME3', 'ME4', 'BIG']):
        for j, bm in enumerate(['LoBM', 'BM2', 'BM3', 'BM4', 'HiBM']):
            if size == 'SMALL' and bm == 'LoBM':
                ff25_names.append('SMALLLoBM')
                ff25_descriptions.append('Small Growth')
            elif size == 'SMALL' and bm == 'HiBM':
                ff25_names.append('SMALLHiBM') 
                ff25_descriptions.append('Small Value')
            elif size == 'BIG' and bm == 'LoBM':
                ff25_names.append('BIGLoBM')
                ff25_descriptions.append('Large Growth')
            elif size == 'BIG' and bm == 'HiBM':
                ff25_names.append('BIGHiBM')
                ff25_descriptions.append('Large Value')
            else:
                # Standard naming
                portfolio_name = f"{size}{bm}"
                portfolio_desc = f"{size_labels[i]} {bm_labels[j]}"
                ff25_names.append(portfolio_name)
                ff25_descriptions.append(portfolio_desc)
    
    # Create FF25 mapping
    map_ff25 = dict(zip(ff25_names, ff25_descriptions))
    
    # Return lags
    nlags = 60
    retlags_names = [f'ret_lag{i}' for i in range(1, nlags+1)]
    retlags_descriptions = [f'Month $t-{i}$' for i in range(1, nlags+1)]
    map_retlags = dict(zip(retlags_names, retlags_descriptions))
    
    # PCs
    npc = 100
    pc_names = [f'PC{i}' for i in range(1, npc+1)]
    pc_descriptions = [f'PC {i}' for i in range(1, npc+1)]
    map_pcs = dict(zip(pc_names, pc_descriptions))
    
    # Combine all mappings
    combined_map = {}
    combined_map.update(map_anomalies)
    combined_map.update(map_ff25)
    combined_map.update(map_retlags)
    combined_map.update(map_pcs)
    
    return combined_map