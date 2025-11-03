#!/usr/bin/env python3
"""
Output Helper Module for SREINet
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def save_sreinet_results_to_excel(config_df, training_histories, recovered_model, 
                                 filename=None, output_dir='output'):
    """
    Save SREINet results to Excel file with multiple sheets.    
    """
    
    # Generate filename with timestamp
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sreinet_results_{timestamp}.xlsx"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    
    # Create Excel writer
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        
        # Sheet 1: Configuration
        config_df.to_excel(writer, sheet_name='configuration', index=False)
        
        # Sheet 2: Training History (if available)
        if training_histories and len(training_histories) > 0:
            
            # Find maximum epochs across all dimensions
            max_epochs = max(len(h['loss']) for h in training_histories if h and 'loss' in h)
            
            # Create combined training history with all dimensions in columns
            combined_data = {'Epoch': list(range(1, max_epochs + 1))}
            
            # Add each dimension's loss and val_loss as separate columns
            for i, history in enumerate(training_histories):
                if history and 'loss' in history and 'val_loss' in history:
                    # Pad with NaN if this dimension has fewer epochs
                    train_losses = history['loss'] + [np.nan] * (max_epochs - len(history['loss']))
                    val_losses = history['val_loss'] + [np.nan] * (max_epochs - len(history['val_loss']))
                    
                    combined_data[f'Dim_{i+1}_Train_Loss'] = train_losses
                    combined_data[f'Dim_{i+1}_Val_Loss'] = val_losses
            
            # Save combined training history
            combined_df = pd.DataFrame(combined_data)
            combined_df.to_excel(writer, sheet_name='Training_History', index=False)
            
        
        # Sheet 3: Recovered Model 
        if recovered_model and hasattr(recovered_model, 'export_equations'):
            eq_str = recovered_model.export_equations()
            if eq_str:
                recovered_df = pd.DataFrame(eq_str, columns=['Recovered_Equations'])
                recovered_df.to_excel(writer, sheet_name='recovered_model', index=False)
    
    print(f"Successfully saved SREINet results to: {filepath}")
    return filepath

