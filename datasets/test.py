import pandas as pd
import glob

files = glob.glob("**/*.parquet", recursive=True)
for f in files:
    df = pd.read_parquet(f)
    print(f"{f}: {len(df)} rows, columns: {df.columns.tolist()}")
