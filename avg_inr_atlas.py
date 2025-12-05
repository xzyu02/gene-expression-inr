import nibabel as nib
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Average INR Atlas processing")
    parser.add_argument('--mode', type=str, choices=['avg_atlas', 'avg_nii'], default='avg_atlas', help='Operation mode')
    parser.add_argument('--matter', type=str, default='83_new', help='Matter type (e.g., 83_new)')
    parser.add_argument('--donor', type=str, default='9861', help='Donor ID')
    parser.add_argument('--atlas', type=str, default='atlas-desikankilliany', help='Atlas name')
    parser.add_argument('--all_records', action='store_true', help='Use full records naming convention (inrs vs inr)')
    parser.add_argument('--gene_list', type=str, help='Path to gene list CSV')
    parser.add_argument('--input_dir', type=str, help='Input directory for avg_atlas mode')
    
    # Arguments for avg_nii
    parser.add_argument('--nii1_dir', type=str, help='First directory for NIfTI files (avg_nii mode)')
    parser.add_argument('--nii2_dir', type=str, help='Second directory for NIfTI files (avg_nii mode)')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    
    return parser.parse_args()

def avg_atlas(gene_atlas_path, atlas, donor, matter, input_dir, output_dir):
    input_path = os.path.join(input_dir, f'{gene_atlas_path}.nii.gz')
    output_path = os.path.join(output_dir, f'{gene_atlas_path}_avg.nii.gz')
    
    atlas_with_labels = nib.load(f'./data/atlas/{atlas}.nii.gz')
    atlas_with_genes = nib.load(input_path)

    # Get data from images
    label_data = atlas_with_labels.get_fdata()
    gene_data = atlas_with_genes.get_fdata()

    regions = np.unique(label_data)
    label_voxel_coordinates = {label: np.argwhere(label_data == label) for label in regions}
    
    new_gene_data = np.zeros_like(gene_data)
    avg_map = {}
        
    for label, coords in label_voxel_coordinates.items():
        # get average value of coords in atlas_with_gene
        if label == 0:
            continue
        
        gene_values = [gene_data[tuple(coord)] for coord in coords]    
        avg_gene_expression = np.mean(gene_values)
        
        new_gene_data[tuple(zip(*coords))] = avg_gene_expression
        avg_map[int(label)] = avg_gene_expression
    
    new_img = nib.Nifti1Image(new_gene_data, atlas_with_genes.affine)
    nib.save(new_img, output_path)
    # print(f"Interpolated {output_path} Success!")
    return avg_map

def average_2_nii_files(file1, file2, output_file):
    nii1 = nib.load(file1)
    nii2 = nib.load(file2)

    data1 = nii1.get_fdata()
    data2 = nii2.get_fdata()

    # Check if the shapes of the two data arrays are the same
    if data1.shape != data2.shape:
        raise ValueError("The input NIfTI files must have the same shape.")

    # Compute the average of the two data arrays
    avg_data = (data1 + data2) / 2

    # Create a new NIfTI image
    avg_img = nib.Nifti1Image(avg_data, affine=nii1.affine, header=nii1.header)

    # Save the new NIfTI image
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(avg_img, output_file)

def main():
    args = parse_args()

    if not args.gene_list:
        raise ValueError("Please provide --gene_list")
    df = pd.read_csv(args.gene_list)
    
    if args.mode == 'avg_atlas':
        if not args.input_dir:
            raise ValueError("For avg_atlas mode, please provide --input_dir")
        input_dir = args.input_dir
        output_dir = args.output_dir if args.output_dir else input_dir
        
        os.makedirs(output_dir, exist_ok=True)
        genes_data = {}

        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            id = row['gene_symbol']
            path = f"{id}_{args.matter}_inr"
                
            avg_map = avg_atlas(path, args.atlas, args.donor, args.matter, input_dir=input_dir, output_dir=output_dir)
            genes_data[id] = avg_map
            
        # Save results
        df_res = pd.DataFrame(genes_data)
        df_res.index.name = 'label'
        df_res = df_res.sort_index(axis=1)
        
        suffix = "inrs" if args.all_records else "inr"
        output_csv = os.path.join(output_dir, f'result_{args.matter}_{args.donor}_{suffix}_avg.csv')

        df_res.to_csv(output_csv)
        print(f"Saved average atlas data to {output_csv}")

    elif args.mode == 'avg_nii':
        if not args.nii1_dir or not args.nii2_dir or not args.output_dir:
            print("Error: For avg_nii mode, please provide --nii1_dir, --nii2_dir, and --output_dir")
            return

        os.makedirs(args.output_dir, exist_ok=True)
        
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            id = row['gene_symbol']
            # Construct filename based on convention
            filename = f"{id}_{args.matter}_inr.nii.gz"
            
            file1 = os.path.join(args.nii1_dir, filename)
            file2 = os.path.join(args.nii2_dir, filename)
            output_file = os.path.join(args.output_dir, filename)
            
            average_2_nii_files(file1, file2, output_file)

if __name__ == "__main__":
    main()