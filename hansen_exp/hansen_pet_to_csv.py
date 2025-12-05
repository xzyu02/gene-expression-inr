import pandas as pd
import numpy as np
import os
from nilearn.maskers import NiftiLabelsMasker

# please pull from https://github.com/netneurolab/hansen_receptors/tree/main/data/PET_nifti_images

pet_folder = '/oscar/data/tserre/xyu110/hansen_receptors/data/PET_nifti_images'

atlas_path = 'data/atlas/atlas-desikankilliany.nii.gz'
gene_list_csv = 'data/hansen_2022_full.csv'
output_csv = 'data/hansen_pet_values_83regions.csv'

# Genes to explicitly exclude (Missing in AHBA due to mRNA/Protein mismatch)
excluded_genes = ['SLC6A2', 'SLC6A4', 'HTR6', 'HRH3', 'HTR1B', 'SLC18A3'] 

print(f"Loading gene map from: {gene_list_csv}")
try:
    gene_df = pd.read_csv(gene_list_csv)
    
    # Filter: You can uncomment the line below if you ONLY want the "Recommended" set
    # gene_df = gene_df[gene_df['Validation_Set_Recommended'] == 'Yes']
    
    # Create dictionary: {Gene_Symbol: Protein_Name}
    # e.g., {'DRD2': 'D2', 'HTR1A': '5-HT1A'}
    full_gene_map = pd.Series(
        gene_df.Protein_Name_In_Paper.values, 
        index=gene_df.Gene_Symbol
    ).to_dict()
    
    # Remove excluded genes
    gene_map = {k: v for k, v in full_gene_map.items() if k not in excluded_genes}
    
    print(f"Found {len(full_gene_map)} genes. Processing {len(gene_map)} valid targets (excluded {len(excluded_genes)} known missing).")

except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# Manual mapping of Protein Name to specific filename in the PET folder
filename_mapping = {
    'D1': 'D1_SCH23390_hc13_kaller.nii',
    'D2': 'D2_fallypride_hc49_jaworska.nii',
    'DAT': 'DAT_fpcit_hc174_dukart_spect.nii',
    '5-HT1A': '5HT1a_way_hc36_savli.nii',
    '5-HT2A': '5HT2a_cimbi_hc29_beliveau.nii',
    '5-HT4': '5HT4_sb20_hc59_beliveau.nii',
    'a4b2': 'A4B2_flubatine_hc30_hillmer.nii.gz',
    'M1': 'M1_lsn_hc24_naganawa.nii.gz',
    'NMDA': 'NMDA_ge179_hc29_galovic.nii.gz',
    'mGluR5': 'mGluR5_abp_hc73_smart.nii',
    'GABA-A/BZ': 'GABAa-bz_flumazenil_hc16_norgaard.nii',
    'CB1': 'CB1_omar_hc77_normandin.nii.gz',
    'MOR': 'MU_carfentanil_hc204_kantonen.nii'
}

# NiftiLabelsMasker handles the resampling of PET images to match the atlas
masker = NiftiLabelsMasker(labels_img=atlas_path, 
                           standardize=False,  # Keep raw density values
                           resampling_target='labels',
                           verbose=1)
regional_data = {}

print("Starting extraction...")

for gene_symbol, protein_name in gene_map.items():
    # Construct filename: usually "ProteinName.nii"
    # We check for both .nii and .nii.gz
    base_name = protein_name
    
    target_file = None
    
    if protein_name in filename_mapping:
        mapped_name = filename_mapping[protein_name]
        mapped_path = os.path.join(pet_folder, mapped_name)
        if os.path.exists(mapped_path):
            target_file = mapped_path
        
            
    if target_file:
        print(f"Processing {gene_symbol} ({protein_name}) -> found {os.path.basename(target_file)}")
        try:
            # Extract average signal per region
            # returns shape (1, n_regions)
            signals = masker.fit_transform(target_file)
            
            # Flatten to 1D array
            regional_data[gene_symbol] = signals.flatten()
            
        except Exception as e:
            print(f"!! Error extracting {gene_symbol}: {e}")
    else:
        print(f"!! File not found for {gene_symbol} (Expected {base_name}.nii/.gz)")


if regional_data:
    # Convert to DataFrame: Rows=Genes, Cols=Regions
    df_out = pd.DataFrame.from_dict(regional_data, orient='index')
    
    # Sort by gene symbol
    df_out.sort_index(inplace=True)
    
    # Rename columns to Region_0, Region_1...
    df_out.columns = [f"Region_{i}" for i in range(df_out.shape[1])]
    
    print(f"\nExtraction complete.")
    print(f"Matrix shape: {df_out.shape} ({df_out.shape[0]} Genes x {df_out.shape[1]} Regions)")
    
    df_out.to_csv(output_csv)
    print(f"Saved results to: {output_csv}")
else:
    print("No data extracted. Check paths and filenames.")