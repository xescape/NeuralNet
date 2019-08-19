from pathlib import Path
import pandas as pd
from numpy import sum as npsum 

def count_nocall(row):
    msk = row == '.'
    s = npsum(msk)
    return s

if __name__ == "__main__":
    
    dir = Path('/d/data/plasmo/training_data')
    table_path = dir / 'nocall_filtered.tsv'
    out_path = dir / 'nocallbysample.tsv'
    table_out_path = dir / 'sample_filtered.tsv'

    df = pd.read_csv(table_path, sep='\t', header=0) 

    counts = df.apply(count_nocall, axis = 0)
    counts.index = list(df.columns)
    counts.sort_values(ascending=False, inplace=True)
    counts.to_csv(out_path, sep='\t', header=False)
    
    to_drop = list(counts.index)[:10]
    df_filtered = df.drop(columns=to_drop)
    df_filtered.to_csv(table_out_path, sep='\t', header=True, index=False)


    