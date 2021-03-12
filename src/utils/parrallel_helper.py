from joblib import Parallel, delayed
import pandas as pd
import multiprocessing as mp
from log_helper import *


MAX_JOBS = int(mp.cpu_count()) 

g_log_helper = LogHelper(log_name='test.log')
log = g_log_helper.make_logger(level=logbook.INFO)


def apply_parallel_helper(func, df:pd.DataFrame, params, *args, **kwargs):
    res = []
    for index, item in df.iterrows():
        # print( item[params] )
        res.append( func( item[params], *args, **kwargs ))
    
    return res


def apply_parallel(func, data:pd.DataFrame, params='id', n_jobs = MAX_JOBS, verbose=0, *args, **kwargs):
    if data.shape[0] < n_jobs:
        n_jobs = data.shape[0]
        
    data.loc[:,'group'] = data.index % n_jobs
    df = data.groupby('group')
    
    results = Parallel(
        n_jobs=n_jobs, verbose=verbose)(
            delayed(apply_parallel_helper)(func, group, params, *args, **kwargs) for name, group in df 
        )
    
    return results


def fake_func(id, *args, **kwargs):
    sum = 0
    for i in range(10**id):
        sum += i
    log.info(f"{id}, {sum}")
    return sum


if __name__ == '__main__':
    df = pd.DataFrame( range(20), columns=['id'] )
    res = apply_parallel( fake_func, df, "id", verbose=0 )
    print(res)
    
    # apply_parallel_helper( fake_func, df, 'id' )
