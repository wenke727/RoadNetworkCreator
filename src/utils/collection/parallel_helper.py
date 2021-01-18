from joblib import Parallel, delayed
import pandas as pd

def apply_parallel(df, func, n_jobs = 52):
    df.loc[:,'group'] = df.index % n_jobs
    df = df.groupby('group')
    results = Parallel(n_jobs=n_jobs)(delayed(func)(group) for name, group in df)
    print("Done!")
    return pd.concat(results)


