import numpy as np
import pandas as pd
import nibabel as nib
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# --- Configuration ---
INPUT_MATTER = "83_new" # Prefix for microarray files in data/abagendata/ (e.g. 83_microarray_...)
ANNOTATION_DIR = "./data/abagendata/train_83_new" # Directory containing 83_annotation_{donor}_4d.csv
DONORS = ["9861", "10021"]
# ---------------------

def mni2vox(mni_coords, affine):
    voxel_coords = np.linalg.inv(affine) @ np.append(mni_coords, 1)
    return np.rint(voxel_coords[:3]).astype(int)

def generate4d(filename):
    # Load the NIfTI files for brain, white matter, and grey matter
    # Assuming script is run from project root
    brain = nib.load("./data/atlas/MNI152_T1_1mm_brain.nii.gz")
    white_matter = nib.load('./data/atlas/MNI152_T1_1mm_brain_white_mask.nii.gz')
    grey_matter = nib.load('./data/atlas/MNI152_T1_1mm_brain_grey_mask.nii.gz')

    # Get the data arrays for each component
    brain_data = brain.get_fdata()
    white_data = white_matter.get_fdata()
    grey_data = grey_matter.get_fdata()
        
    # Function to check if a point is in white matter, grey matter, or neither
    def check_matter_type(voxel_coord, white_data, grey_data):
        x, y, z = voxel_coord
        if white_data[int(x), int(y), int(z)] > 0:
            return 1  # Indicates white matter
        elif grey_data[int(x), int(y), int(z)] > 0:
            return -1  # Indicates grey matter
        else:
            return 0  # Indicates neither

    meta_df = pd.read_csv(f"{filename}.csv")
    coords = meta_df[['mni_x', 'mni_y', 'mni_z']].values

    classifications = []
    for mni_coord in coords:
        voxel_coord = mni2vox(mni_coord, brain.affine)
        classification = check_matter_type(voxel_coord, white_data, grey_data)
        classifications.append(classification)

    meta_df['classification'] = classifications
    meta_df.to_csv(f"{filename}_4d.csv", index=False)
    print(f"Saved {filename}_4d.csv")

def get_pc1_se_results(donor, input_matter, output_dir, gene_list):
    gene_names = set(gene_list)

    # Read microarray data from the common location
    microarray_path = f"./data/abagendata/{input_matter}_microarray_{donor}.csv"
    print(f"Reading microarray data from: {microarray_path}")
    microarray_df = pd.read_csv(microarray_path)
    microarray_df = microarray_df.iloc[:, 1:].T
    microarray_df.columns = microarray_df.iloc[0].astype(int).astype(str)
    microarray_df = microarray_df.iloc[1:]
    microarray_df.index.name = 'gene_symbol'

    # find gene names not in microarry_df
    missing_genes = gene_names - set(microarray_df.index)
    if missing_genes:
        print(f"Donor {donor} - Missing genes ({len(missing_genes)}): ", list(missing_genes)[:10], "...")
    
    gene_names = gene_names - missing_genes
    # use gene_names to get a gene dataframe
    gene_names = list(gene_names)
    gene_df = microarray_df.loc[gene_names]

    # PCA
    gene_df = gene_df.T
    pca = PCA(n_components=1)
    pca.fit(gene_df)
    pc1_loadings = pca.components_[0]

    gene_df_pca = gene_df.T
    # add pc1_loadings to gene_df as a new column
    gene_df_pca["pc1"] = pc1_loadings
    # sort gene_df by pc1
    gene_df_pca = gene_df_pca.sort_values(by="pc1", ascending=True)
    
    # save gene_df to a new csv file in the output directory
    pc1_output_path = os.path.join(output_dir, f"pc1_{donor}.csv")
    gene_df_pca.to_csv(pc1_output_path)
    print(f"Saved {pc1_output_path}")


    # Spectrum Embedding
    gene_df_embedding = gene_df.T
    embedding = SpectralEmbedding(n_components=1)
    gene_embedding = embedding.fit_transform(gene_df_embedding)

    gene_df_embedding["se"] = gene_embedding[:, 0].flatten()
    gene_df_embedding = gene_df_embedding.sort_values(by="se", ascending=True)
    
    scaler = MinMaxScaler()
    gene_df_embedding['se'] = scaler.fit_transform(gene_df_embedding[['se']])
    
    se_output_path = os.path.join(output_dir, f"se_{donor}.csv")
    gene_df_embedding.to_csv(se_output_path)
    print(f"Saved {se_output_path}")

def merge_5d(mode, donor, output_dir, annotation_dir, matter_prefix="83"):
    # Read the generated data from output_dir
    df_path = os.path.join(output_dir, f"{mode}_{donor}.csv")
    df = pd.read_csv(df_path)
    
    # Read the annotation from the annotation_dir
    annot_path = os.path.join(annotation_dir, f"{matter_prefix}_annotation_{donor}_4d.csv")
    print(f"Reading annotation from: {annot_path}")
    annot = pd.read_csv(annot_path)
    
    df_long = df.melt(id_vars=[mode, 'gene_symbol'], var_name='well_id', value_name='value')
    annot = annot[["mni_x", "mni_y", "mni_z", "classification", "well_id"]]
    
    df_long['well_id'] = df_long['well_id'].astype(str)
    annot['well_id'] = annot['well_id'].astype(str)
    
    print(f"Merging {mode} for donor {donor}: df shape {df_long.shape}, annot shape {annot.shape}")

    merged_df = pd.merge(df_long, annot, on='well_id', how='inner')
    
    merged_df = merged_df[['gene_symbol', 'well_id', 'mni_x', 'mni_y', 'mni_z', 'classification', 'value', mode]]
    merged_df = merged_df.sort_values(by=['gene_symbol', 'well_id'])
    
    merged_output_path = os.path.join(output_dir, f"{mode}_{donor}_merged.csv")
    merged_df.to_csv(merged_output_path, index=False)
    print(f"Saved {merged_output_path}")

if __name__ == "__main__":
    # Load gene lists from Hansen 2022
    print("Loading gene lists from ./data/hansen_2022_full.csv...")
    hansen_df = pd.read_csv("./data/hansen_2022_full.csv")
    
    full_genes = hansen_df['Gene_Symbol'].unique().tolist()
    recommended_genes = hansen_df[hansen_df['Validation_Set_Recommended'] == 'Yes']['Gene_Symbol'].unique().tolist()
    
    # Load common gene list
    print("Loading common gene list from ./data/common_gene_list.csv...")
    with open("./data/common_gene_list.csv", "r") as f:
        common_genes = [line.strip() for line in f if line.strip()]

    print(f"Found {len(full_genes)} total genes in Hansen.")
    print(f"Found {len(recommended_genes)} recommended genes in Hansen.")
    print(f"Found {len(common_genes)} genes in common_gene_list.")

    # Union with common_gene_list
    full_genes = list(set(full_genes).union(set(common_genes)))
    recommended_genes = list(set(recommended_genes).union(set(common_genes)))
    
    print(f"After union with common_gene_list:")
    print(f"  Full genes: {len(full_genes)}")
    print(f"  Recommended genes: {len(recommended_genes)}")

    experiments = [
        ("hansen_full", full_genes),
        ("hansen_recommended", recommended_genes)
    ]

    for exp_name, genes in experiments:
        output_dir = f"./data/abagendata/train_{exp_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n==================================================")
        print(f"--- Running Experiment: {exp_name} ---")
        print(f"Output Directory: {output_dir}")
        print(f"Number of genes: {len(genes)}")
        print(f"==================================================\n")

        for donor in DONORS:            
            # 1. PC1 and SE
            get_pc1_se_results(donor, INPUT_MATTER, output_dir, genes)
            
            # 2. Merge (using existing annotation from ANNOTATION_DIR)
            merge_5d('se', donor, output_dir, ANNOTATION_DIR, matter_prefix="83")
        
    print("\nAll processing complete.")
