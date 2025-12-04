import argparse
import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.my_modules import *
from modules import models

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for Gene Expression INR")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--nonlin', type=str, default='siren', help='Non-linearity type (e.g., oinr2d, siren)')
    parser.add_argument('--hidden_features', type=int, default=512, help='Number of hidden features')
    parser.add_argument('--hidden_layers', type=int, default=12, help='Number of hidden layers')
    parser.add_argument('--in_features', type=int, default=27, help='Number of input features (default 27 for all_records)')
    
    # Data parameters
    parser.add_argument('--atlas', type=str, default='atlas-desikankilliany', help='Atlas name')
    parser.add_argument('--matter', type=str, default='83_new', help='Matter type (e.g., 83_new, grey)')
    parser.add_argument('--donor', type=str, default='9861', help='Donor ID')
    parser.add_argument('--dataset', type=str, default='hansen_recommended', help='Dataset type (e.g., hansen_recommended, hansen_full)')
    
    # Min/Max values CSV
    parser.add_argument('--min_max_path', type=str, default='./hansen_ckpts/max_min_values_se_sep.csv', help='Path to min/max values CSV')
    
    # Gene selection
    parser.add_argument('--gene_list', type=str, help='Path to CSV containing gene list (e.g., se_{donor}.csv)')
    parser.add_argument('--gene_id', type=str, help='Single gene ID to run inference for')
    parser.add_argument('--se_val', type=float, help='SE value for single gene inference')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save output NIfTI files')
    
    # Flags
    parser.add_argument('--all_records', action='store_true', help='Use all_records mode (positional encoding)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = f'./{args.nonlin}_nii_output'
    return args

def load_model(args, device, model_path):
    # Determine in_features based on all_records flag if not explicitly set? 
    # The user provided in_features default 27, but inference.py uses 5 for !all_records.
    # I'll stick to args.in_features but maybe adjust default in main if needed.
    
    if args.all_records:
        model = models.get_INR(
            nonlin=args.nonlin,
            in_features=args.in_features,
            out_features=1,
            hidden_features=args.hidden_features,
            hidden_layers=args.hidden_layers,
            scale=5.0,
            pos_encode=False,
            sidelength=256)
    else:
        # Fallback to Siren if not all_records, as in original inference.py?
        model = models.get_INR(
            nonlin=args.nonlin if args.nonlin else 'siren',
            in_features=args.in_features if args.in_features != 27 else 5, # Adjust default if not changed
            out_features=1,
            hidden_features=256, # Original code used 256 for !all_records
            hidden_layers=5,     # Original code used 5 for !all_records
            scale=5.0, # Not used in Siren directly but passed to get_INR
            pos_encode=False,
            sidelength=256
        )
        
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model

def get_result(id, xyz, model, args, device, min_max_dict_df):
    # Filter
    if args.all_records:
        query_id = 'ALL_RECORDS'
    else:
        query_id = str(id)
        
    # Filter by ID
    df_filtered = min_max_dict_df[min_max_dict_df['id'] == query_id]
    
    # Filter by donor if column exists and not ALL_RECORDS (or even for ALL_RECORDS if applicable)
    if 'donor' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['donor'] == int(args.donor)]
        
    # Filter by dataset if column exists
    if 'dataset' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['dataset'] == args.dataset]
        
    if df_filtered.empty:
        print(f"Warning: No records found for id={query_id}, donor={args.donor}, dataset={args.dataset}. Using default/last available.")
        # Fallback logic: try to find by ID only
        df_filtered = min_max_dict_df[min_max_dict_df['id'] == query_id]
    
    records = df_filtered.to_dict(orient='records')
    if not records:
        raise ValueError(f"No records found for id {query_id} in min/max CSV")
        
    if len(records) > 1:
        # print(f"Warning: Found {len(records)} records. Using the last one.")
        pass
    min_max_dict = records[-1]

    min_vals = torch.tensor(min_max_dict['min_vals'])
    max_vals = torch.tensor(min_max_dict['max_vals'])
    min_coords = torch.tensor(min_max_dict['min_coords'])
    max_coords = torch.tensor(min_max_dict['max_coords'])
    
    def normalize_coord(coords):
        coords_to_normalize = coords[:, :3]
        coords_fixed = coords[:, 3:]
        
        normalized_coords = (coords_to_normalize - min_coords) * (1 - (-1)) / (max_coords - min_coords) - 1
        return torch.cat((normalized_coords, coords_fixed), dim=1)
    
    coords = torch.tensor(xyz, dtype=torch.float32).to(device)
    coords = normalize_coord(coords)
    coords = coords.unsqueeze(0)
    
    with torch.no_grad():
        output = model(coords)
        # Handle output format (tuple or tensor)
        if isinstance(output, tuple):
            output = output[0]
        result = output.cpu().detach().numpy().flatten()
    
    return result

def get_results(id, xyz, model, args, device, min_max_dict_df, batch_size=2**18):
    results = []
    for i in range(0, len(xyz), batch_size):
        batch_xyz = xyz[i:i+batch_size]
        batch_results = get_result(id, batch_xyz, model, args, device, min_max_dict_df)
        batch_results = np.asarray(batch_results).flatten()
        results.extend(batch_results)
    return np.array(results)

def inference(id, order_val, args, model, device, min_max_dict_df):
    nii_file = f'./data/atlas/{args.atlas}.nii.gz'
    if not os.path.exists(nii_file):
        print(f"Error: Atlas file not found at {nii_file}")
        return

    image = nib.load(nii_file)
    data = image.get_fdata()
    affine = image.affine
    
    x_dim, y_dim, z_dim = data.shape

    xyz = []
    mni_coords = []
    # Optimize this loop?
    # Using numpy indices is faster
    indices = np.argwhere(data > 0)
    xyz = indices
    # Vectorized vox2mni
    # vox2mni: affine @ [x,y,z,1]
    # We can do this in batch
    ones = np.ones((indices.shape[0], 1))
    indices_homo = np.hstack((indices, ones))
    mni_coords = (affine @ indices_homo.T).T[:, :3]

    if args.matter == "white":
        classification_val = 1
    elif args.matter in ["grey", "246", "83", "83_new"]:
        classification_val = -1
    else:
        print("Error: Unknown brain matter type")
        return
        
    # Prepare meta_df
    meta_df = pd.DataFrame(mni_coords, columns=['mni_x', 'mni_y', 'mni_z'])
    meta_df['classification'] = classification_val
    meta_df['se'] = order_val
    
    if args.all_records:
        encoding_dim = 11
        meta_df = encode_df(meta_df, multires=encoding_dim)

    # print(meta_df.head())
    mni_coords_input = meta_df.to_numpy()

    print(f"Generating Results for gene {id}...")      
    outputs = get_results(id, mni_coords_input, model, args, device, min_max_dict_df)

    plot_data = np.zeros(data.shape)
    
    # Assign outputs back to volume
    # xyz is list of [x,y,z]
    plot_data[xyz[:,0], xyz[:,1], xyz[:,2]] = outputs
        
    new_img = nib.Nifti1Image(plot_data, affine=affine)
    
    save_path = os.path.join(args.output_dir, f'{id}_{args.matter}_inr.nii.gz')
    nib.save(new_img, save_path)
    print(f"Saved to {save_path}")

def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load min/max CSV once
    col_names = ['id', 'min_vals', 'max_vals', 'min_coords', 'max_coords', 'dataset', 'donor']
    min_max_dict_df = pd.read_csv(args.min_max_path, header=None, names=col_names)
    
    # Check if model_path is dynamic
    is_dynamic_model = '{id}' in args.model_path
    
    model = None
    if not is_dynamic_model:
        model = load_model(args, device, args.model_path)
    
    if args.gene_list:
        df = pd.read_csv(args.gene_list)
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            id = row['gene_symbol']
            order_val = row['se']
            
            current_model = model
            if is_dynamic_model:
                current_model_path = args.model_path.format(id=id)
                if os.path.exists(current_model_path):
                    current_model = load_model(args, device, current_model_path)
                else:
                    print(f"Model not found: {current_model_path}")
                    continue
            
            inference(id, order_val, args, current_model, device, min_max_dict_df)
    elif args.gene_id:
        if args.se_val is None:
            raise ValueError("Must provide --se_val when using --gene_id")
            
        current_model = model
        if is_dynamic_model:
            current_model_path = args.model_path.format(id=args.gene_id)
            if os.path.exists(current_model_path):
                current_model = load_model(args, device, current_model_path)
            else:
                print(f"Model not found: {current_model_path}")
                return

        inference(args.gene_id, args.se_val, args, current_model, device, min_max_dict_df)
    else:
        print("Error: Must provide either --gene_list or --gene_id")

if __name__ == "__main__":
    main()
