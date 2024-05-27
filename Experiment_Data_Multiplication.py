import pandas as pd
import numpy as np
import os

# Define augmentation functions
def add_noise(spectrum, intensity):
    noise = intensity * np.random.normal(size=len(spectrum))
    return np.maximum(spectrum + noise, 0)  # Ensure no negative values

def multiply_by_coefficient(spectrum, num_variants):
    coefficients = np.linspace(0.1, 0.9, num=num_variants)
    augmented_spectra = [np.maximum(spectrum * coef, 0) for coef in coefficients]
    return augmented_spectra

def power_transform(spectrum, num_variants):
    powers = np.linspace(1.1, 2.0, num=num_variants)
    augmented_spectra = [np.maximum(np.power(spectrum, power), 0) for power in powers]
    return augmented_spectra

# Configuration for each experiment type
augmentation_config = {
    'experiment_1': {'method': add_noise, 'params': {'intensity': 0.05}, 'name': 'noise_augmentation'},
    'experiment_2': {'method': multiply_by_coefficient, 'params': {}, 'name': 'coefficient_augmentation'},
    'experiment_3': {'method': power_transform, 'params': {}, 'name': 'power_augmentation'}
}

def augment_data(file_path, config_key, num_variants=100, output_dir=""):
    print(f"Config Key: {config_key}")  # Debug
    print(f"Available Configurations: {augmentation_config}")  # Debug
    
    data = pd.read_excel(file_path)
    data = data.apply(pd.to_numeric, errors='coerce')
    spectra = data.iloc[:, :]
    augmented_data = []

    for _, row in spectra.iterrows():
        method = augmentation_config[config_key]['method']
        params = augmentation_config[config_key]['params']
        
        if config_key in ['experiment_2', 'experiment_3']:
            # For coefficient and power augmentations
            augmented_spectra = method(row.values, num_variants)
            augmented_data.extend(augmented_spectra)
        else:
            # For noise augmentation
            augmented_spectrum = method(row.values, **params)
            for _ in range(num_variants):
                augmented_data.append(augmented_spectrum)

    augmented_df = pd.DataFrame(augmented_data)
    augmentation_name = augmentation_config[config_key]['name']
    output_file = os.path.join(output_dir, f"{augmentation_name}_{os.path.basename(file_path).replace('.xlsx', '.csv')}")

    os.makedirs(output_dir, exist_ok=True)
    augmented_df.to_csv(output_file, index=False, header=False)
    print(f"File saved to {output_file}")
    
    # Check the augmented data for negative values
    augmented_df = pd.read_csv(output_file)
    if (augmented_df < 0).any().any():
        print("There are negative values in the augmented data.")
    else:
        print("No negative values in the augmented data.")

# Example usage
if __name__ == "__main__":
    output_path = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\UV\Code\code_files\UV Spectrum\Data train\17_05 data"
    file_path = r"C:\Users\thaim\OneDrive\Desktop\Tal_Projects\Gas_detector\General_Codes\new tests\1m_ammonia.xlsx"
    augment_data(file_path, 'experiment_3', output_dir=output_path)
