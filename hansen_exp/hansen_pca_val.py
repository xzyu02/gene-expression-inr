import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
pet_file = 'data/hansen_pet_values_83regions.csv'

# Dictionary of all INR result files to process
inr_files = {
    'Full_9861':   'results/hansen_full_9861/result_83_new_9861_inrs_avg.csv',
    'Full_10021':  'results/hansen_full_10021/result_83_new_10021_inrs_avg.csv',
    'Rec_9861':    'results/hansen_recommended_9861/result_83_new_9861_inrs_avg.csv',
    'Rec_10021':   'results/hansen_recommended_10021/result_83_new_10021_inrs_avg.csv'
}

# Define groups: Key = PET Map Name, Value = List of Genes
pca_groups = {
    'GRIN1':  ['GRIN1', 'GRIN2A', 'GRIN2B'],                     # NMDA Complex
    'GABRA1': ['GABRA1', 'GABRA2', 'GABRB1', 'GABRB2', 'GABRG2'], # GABA-A Complex
    'CHRNA4': ['CHRNA4', 'CHRNB2'],                              # Nicotinic ACh
    'DRD2':   ['DRD2', 'DRD3'],                                  # D2-like Family
    'HTR2A':  ['HTR2A', 'HTR2C']                                 # 5-HT2 Family
}

# ==========================================
# PROCESSING FUNCTIONS
# ==========================================
def load_pet_data(pet_path):
    print(f"Loading PET data from {pet_path}...")
    pet = pd.read_csv(pet_path, index_col=0)
    # Reset columns to simple integer index (0..82) to ensure alignment
    pet.columns = range(pet.shape[1])
    return pet

def process_inr_file(file_path):
    if not os.path.exists(file_path):
        print(f"!! File not found: {file_path}")
        return None
        
    inr = pd.read_csv(file_path, index_col=0)
    
    # Transpose if needed (we need Rows=Genes, Cols=Regions for initial selection)
    # Assuming regions are columns ~83
    if inr.shape[0] == 83: 
        inr = inr.T
        
    # Reset columns to match PET
    inr.columns = range(inr.shape[1])
    return inr

def run_pca_validation(pet_df, inr_df, groups, experiment_name):
    experiment_results = []
    
    for target, genes in groups.items():
        # 1. Validation checks
        if target not in pet_df.index:
            # print(f"  [Skipping {target}] Target PET map not found.")
            continue
            
        valid_genes = [g for g in genes if g in inr_df.index]
        if len(valid_genes) < 2:
            # print(f"  [Skipping {target}] Not enough genes found in INR (Found: {valid_genes})")
            continue

        # 2. Extract Data for PCA
        # We transpose to (Samples=Regions, Features=Genes) for PCA
        X = inr_df.loc[valid_genes].T.values 
        
        # 3. Run PCA
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X).flatten()
        explained_var = pca.explained_variance_ratio_[0]
        
        # 4. Correlate PC1 with PET Ground Truth
        pet_vector = pet_df.loc[target].values.flatten()
        r, p = pearsonr(pc1, pet_vector)
        
        # PCA sign is arbitrary; flip if negative to capture magnitude
        # Alternatively, check directionality if specific biology dictates it
        if r < 0: 
            r = -r
            
        experiment_results.append({
            'Experiment': experiment_name,
            'Receptor': target,
            'Genes_Used': str(valid_genes),
            'Gene_Count': len(valid_genes),
            'Explained_Var': round(explained_var, 3),
            'PCA_R': round(r, 4),
            'P_Value': round(p, 5)
        })
        
    return experiment_results

# ==========================================
# MAIN EXECUTION
# ==========================================
all_results = []

# 1. Load PET Data once
if os.path.exists(pet_file):
    pet_data = load_pet_data(pet_file)
    
    # 2. Loop through INR files
    print("\n--- Starting Multi-Experiment PCA Validation ---")
    for exp_name, path in inr_files.items():
        print(f"Processing {exp_name}...")
        
        inr_data = process_inr_file(path)
        
        if inr_data is not None:
            # Run validation for this specific experiment
            results = run_pca_validation(pet_data, inr_data, pca_groups, exp_name)
            all_results.extend(results)
            print(f"  -> Generated {len(results)} PCA comparisons.")

    # 3. Output Results
    if all_results:
        df_results = pd.DataFrame(all_results)
        
        print("\n=== Final PCA Validation Summary ===")
        # Display key columns
        print(df_results[['Experiment', 'Receptor', 'Gene_Count', 'Explained_Var', 'PCA_R']].to_string(index=False))
        
        # Save to CSV
        df_results.to_csv('validation_pca_results_all_experiments.csv', index=False)
        print("\nSaved full results to 'validation_pca_results_all_experiments.csv'")
        
        # Optional: Quick Pivot Table for easy comparison
        print("\n=== Comparison Matrix (PCA_R) ===")
        pivot_df = df_results.pivot(index='Receptor', columns='Experiment', values='PCA_R')
        print(pivot_df)
        
    else:
        print("No valid PCA groups processed across any experiment.")
