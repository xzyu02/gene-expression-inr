import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os

# ==========================================
# 1. SETUP FILE PATHS
# ==========================================

# Ground Truth (Hansen PET)
pet_file = 'data/hansen_pet_values_83regions.csv'

# Your INR Result Files
inr_files = {
    'Full_9861':   'results/hansen_full_9861/result_83_new_9861_inrs_avg.csv',
    'Full_10021':  'results/hansen_full_10021/result_83_new_10021_inrs_avg.csv',
    'Rec_9861':    'results/hansen_recommended_9861/result_83_new_9861_inrs_avg.csv',
    'Rec_10021':   'results/hansen_recommended_10021/result_83_new_10021_inrs_avg.csv'
}

# ==========================================
# 2. LOAD GROUND TRUTH
# ==========================================
print(f"Loading PET Ground Truth from: {pet_file}")
# PET Data: Expecting Rows = Genes, Columns = Regions
pet_df = pd.read_csv(pet_file, index_col=0)
print(f"PET Data Shape: {pet_df.shape} (Genes x Regions)")
print(f"PET Genes Found: {pet_df.index.tolist()}\n")

# ==========================================
# 3. VALIDATION LOOP
# ==========================================
results_summary = []

for experiment_name, file_path in inr_files.items():
    print(f"--- Processing {experiment_name} ---")
    
    if not os.path.exists(file_path):
        print(f"!! File not found: {file_path}")
        continue
        
    # Load INR Data
    # You mentioned your data is inverted relative to PET
    inr_df = pd.read_csv(file_path, index_col=0)
    print(f"  > Original Shape: {inr_df.shape}")
    
    # TRANSPOSE STEP: Flip it so Rows=Genes to match PET
    inr_df_t = inr_df.T
    print(f"  > Transposed Shape: {inr_df_t.shape} (Rows=Genes)")
    
    # 1. Find Common Genes (Intersection of Indices)
    common_genes = list(set(pet_df.index) & set(inr_df_t.index))
    
    if not common_genes:
        print("  !! No matching genes found. Check spelling/case in CSV indices.")
        continue
        
    print(f"  > Matching Genes ({len(common_genes)}): {common_genes}")
    
    # 2. Align Regions (Columns)
    # We check if column counts match (83 vs 83)
    if pet_df.shape[1] != inr_df_t.shape[1]:
        print(f"  !! Region mismatch: PET has {pet_df.shape[1]}, INR has {inr_df_t.shape[1]}")
        continue
        
    # Force column names to match for safe slicing (0 to 82)
    pet_subset = pet_df.copy()
    inr_subset = inr_df_t.copy()
    pet_subset.columns = range(pet_subset.shape[1])
    inr_subset.columns = range(inr_subset.shape[1])
    
    # 3. Calculate Correlations
    correlations = []
    for gene in common_genes:
        # Get the two vectors (shape: 83,)
        v_pet = pet_subset.loc[gene].values.flatten().astype(float)
        v_inr = inr_subset.loc[gene].values.flatten().astype(float)
        
        # Pearson Correlation
        r, p = pearsonr(v_inr, v_pet)
        
        correlations.append(r)
        
        results_summary.append({
            'Experiment': experiment_name,
            'Gene': gene,
            'Pearson_R': r,
            'P_Value': p
        })
        
    # Report Average for this Experiment
    avg_r = np.mean(correlations)
    print(f"  > Average Correlation: R = {avg_r:.4f}")
    print("")

# ==========================================
# 4. FINAL REPORT
# ==========================================
if results_summary:
    df_results = pd.DataFrame(results_summary)
    
    print("\n=== FINAL VALIDATION SUMMARY ===")
    
    # 1. Average Performance per Experiment
    print("\nAverage Correlation per Experiment:")
    print(df_results.groupby('Experiment')['Pearson_R'].mean().sort_values(ascending=False))
    
    # 2. Gene-wise Breakdown
    print("\nDetailed Gene-wise Correlations:")
    pivot_df = df_results.pivot(index='Gene', columns='Experiment', values='Pearson_R')
    print(pivot_df)
    
    # Optional: Save to CSV
    pivot_df.to_csv("final_validation_results_matrix.csv")
    print("\nSaved detailed matrix to 'final_validation_results_matrix.csv'")
else:
    print("No results calculated.")