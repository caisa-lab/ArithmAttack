import pandas as pd
from config import DIR_PATH

# Read the CSV files
gsm_df = pd.read_csv(f'{DIR_PATH}/data/gsm/test_preprocessed.csv')
multiArith_df = pd.read_csv(f'{DIR_PATH}/data/multiArith/test_preprocessed.csv')

# Randomly sample 50 rows from each DataFrame
gsm_sample = gsm_df.sample(n=50, random_state=1)
multiArith_sample = multiArith_df.sample(n=50, random_state=1)

# Optionally, save the sampled data to new CSV files
gsm_sample.to_csv(f'{DIR_PATH}/data/gsm/sample_test_preprocessed.csv', index=False)
multiArith_sample.to_csv(f'{DIR_PATH}/data/multiArith/sample_test_preprocessed.csv', index=False)

# Print the sampled data (optional)
print(gsm_sample.head())
print(multiArith_sample.head())
