import os
import pandas as pd

def replace_key_values_in_csvs(root_folder):
    # Iterate through all folders and files in the root folder
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Replace values in the "Key" column
                df['Key'] = df['Key'].replace({'shift_r': 'shift', 'shift_l': 'shift', 'caps_lock': 'shift'})
                
                # Save the modified DataFrame back to CSV
                df.to_csv(file_path, index=False)

# Example usage
replace_key_values_in_csvs('datasets/hai-data/labels')
replace_key_values_in_csvs('datasets/nhi-data/labels')