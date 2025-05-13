import pandas as pd
import torch

# Set random seed for reproducibility
g = torch.Generator()
g.manual_seed(42)

# Read the original parquet file
df = pd.read_parquet('data/nq_hotpotqa_train/train_e5_u1.parquet')

# Get the number of samples
num_samples = len(df)

# Generate shuffled indices using PyTorch's logic
indices = torch.randperm(num_samples, generator=g).tolist()

# Take the first 1000 indices
first_1k_indices = indices[:1000]

# Select those rows from the DataFrame
# (reset index to avoid index mismatch in output file)
df_1k = df.iloc[first_1k_indices].reset_index(drop=True)

# Save to new parquet file
output_path = 'data/nq_hotpotqa_train/train_e5_u1_1K.parquet'
df_1k.to_parquet(output_path, index=False)

print(f"Created new dataset with {len(df_1k)} samples at {output_path}") 