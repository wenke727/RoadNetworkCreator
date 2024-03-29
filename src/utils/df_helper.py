import pandas as pd
import geopandas as gpd

def query_df(df, att, val):
    val = '\''+val+'\'' if isinstance(val, str) else val 
    return df.query( f" {att} == {val} " )


def gdf_concat(lst):
    return gpd.GeoDataFrame(pd.concat(lst))


def load_df_memo(fn=None):
    if fn is None:
        return pd.DataFrame()
    
    df_memo = pd.read_hdf(fn)
    # df_memo.loc[:, 'pred'] = df_memo.pred.apply( lambda x: eval(x) )
    
    return df_memo
