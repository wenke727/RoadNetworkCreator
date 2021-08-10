#%%
import os, sys
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
from shapely.geometry import Point, box

sys.path.append("..") 
from utils.utils import load_config


config   = load_config()
pano_dir = config['data']['pano_dir']
ENGINE   = create_engine(config['data']['DB'])

#%%

def load_postgis(table, bbox=None, geom_wkt=None, engine=ENGINE):
    if bbox is not None:
       geom_wkt = box(*bbox).to_wkt()
     
    if geom_wkt is None:
        sql = f'select * from {table}'
    else:        
        sql = f"""select * from {table} where ST_Within( geometry, ST_GeomFromText('{geom_wkt}', 4326) )"""
        
    df = gpd.read_postgis( sql, geom_col='geometry', con=engine )
    # shenzhen_boundary = gpd.read_file('../input/ShenzhenBoundary_wgs_citylevel.geojson')
    # return gpd.sjoin(df, shenzhen_boundary, op='within')

    return df


def load_from_DB(new=False):
    """load data from DB

    Args:
        new (bool, optional): Create new data or not. Defaults to False.

    Returns:
        (gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame): DB_pano_base, DB_panos, DB_connectors, DB_roads
    """
    if new:
        DB_pano_base, DB_panos, DB_connectors, DB_roads = \
            gpd.GeoDataFrame( crs='EPSG:4326'), gpd.GeoDataFrame(crs='EPSG:4326'),gpd.GeoDataFrame(crs='EPSG:4326'),gpd.GeoDataFrame(crs='EPSG:4326')
        return DB_pano_base, DB_panos, DB_connectors, DB_roads

    DB_pano_base = gpd.read_postgis( 'select * from pano_base', geom_col='geometry', con=ENGINE )
    for att in ['Roads', 'Links']:
        DB_pano_base.loc[:, att] = DB_pano_base[att].apply(lambda x: eval(x))

    DB_panos      = gpd.read_postgis( 'select * from panos', geom_col='geometry', con=ENGINE )
    DB_roads      = gpd.read_postgis( 'select * from roads', geom_col='geometry', con=ENGINE )
    DB_connectors = gpd.read_postgis( 'select * from connectors', geom_col='geometry', con=ENGINE )

    # name each dataframe
    DB_pano_base.name  = 'pano_base'
    DB_panos.name      = 'panos'
    DB_connectors.name = 'connectors'
    DB_roads.name      = 'roads'
    
    return DB_pano_base, DB_panos, DB_connectors, DB_roads


def load_DB_panos():
    return gpd.read_postgis( 'select * from panos', geom_col='geometry', con=ENGINE )


def load_DB_panos_base():
    return gpd.read_postgis( 'select * from pano_base', geom_col='geometry', con=ENGINE )


def load_DB_roads():
    return gpd.read_postgis( 'select * from roads', geom_col='geometry', con=ENGINE )
   

def store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads):
    """
    store road data to DB
    """
    config_local = {"con": ENGINE, 'if_exists':'replace'}

    DB_pano_base_bak = DB_pano_base.copy()
    try:
        # transfer the `list` to `str`, for it not compable for saving in pandas dataframe
        for att in list(DB_pano_base):
            types = DB_pano_base[att].apply(lambda x: type(x)).unique() if att !='geometry' else []
            if list in types:
                DB_pano_base_bak.loc[:, att] = DB_pano_base_bak[att].apply(lambda x: str(x))
                
        DB_pano_base_bak.drop_duplicates(inplace=True)
        DB_pano_base_bak.to_postgis( name='pano_base', **config_local )

        DB_panos.drop_duplicates(inplace=True)
        DB_panos.to_postgis( name='panos', **config_local )
        
        DB_connectors.drop_duplicates(inplace=True)
        DB_connectors.to_postgis( name='connectors', **config_local )
        
        DB_roads.drop_duplicates(inplace=True)
        DB_roads.to_postgis( name='roads', **config_local)

        return True
    except:
        print('Store_to_DB failed!')
        return False


def save_to_db(df, name, config={"con": ENGINE, 'if_exists':'replace'}, verbose=True):
    """Save the GeoDataFrame to the db

    Args:
        df (GeoDataFrame): The dataframe need to be stored in the database.
        name (str): The table name.
        db_config (dict, optional): [description]. Defaults to {"con": ENGINE, 'if_exists':'replace'}.
    """
    try:
        df.to_postgis( name=name, **config)
        return True
    except:
        print('save to db failed!')
        return False
    

def DB_backup(DB_pano_base, DB_panos, DB_connectors, DB_roads):
    """backup the db in the PostgreSQL

    Args:
        DB_pano_base ([type]): [description]
        DB_panos ([type]): [description]
        DB_connectors ([type]): [description]
        DB_roads ([type]): [description]
    """
    config_local = {"con": ENGINE, 'if_exists':'replace'}

    for df_temp in [DB_pano_base, DB_panos,DB_connectors,DB_roads]:
        try:
            if df_temp.name == 'pano_base':
                for att in ['Roads', 'Links']:
                    df_temp.loc[:, att] = DB_pano_base[att].apply(lambda x: str(x))
                    
            df_temp.drop_duplicates(inplace=True)
            df_temp.to_postgis( name=f"back_{df_temp.name}", **config_local )
        except:
            print( f'store dataframe {df_temp.name} failed! ' )


def extract_connectors_from_panos_respond( DB_pano_base, DB_roads ):
    from utils.coord.coord_transfer import bd_mc_to_wgs_vector
    
    # TODO DB_connectors need to be re-constructed 
    roads = DB_pano_base[(DB_pano_base.Links.apply( lambda x: len(x) > 0 )) &
                         (DB_pano_base.ID.isin(DB_roads.PID_end.unique()) )
                        ]

    def _construct_helper(item):
        df = pd.DataFrame(item.Links)
        df.loc[:, 'prev_pano_id'] = item.ID
        return df

    connectors = pd.concat(roads.apply( lambda x: _construct_helper(x), axis=1 ).values.tolist())
    connectors = gpd.GeoDataFrame( connectors, 
                                   geometry = connectors.apply( lambda i: Point( bd_mc_to_wgs_vector(i)), axis=1 ),
                                   crs ='EPSG:4326'
                                 )
    return connectors


def save_to_geojson(gdf, fn):
    if not isinstance(gdf, gpd.GeoDataFrame):
        print('Check the format of the gdf.')
        return False
    
    gdf.to_file(fn, driver='GeoJSON')
    return True


def update_lane_num_in_DB():
    """
    update lane num in panos and roads
    """
    from scipy import stats
    df_memo = pd.read_csv(config['data']['df_pred_memo'])
    df_memo.loc[:, 'pred'] = df_memo.pred.apply( lambda x: eval(x) )
    DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(new = False)

    DB_panos_bak = DB_panos.copy()
    DB_roads_bak = DB_roads.copy()
    # DB_roads = DB_roads_bak.copy()
    # DB_panos = DB_panos_bak.copy()
    
    # update the lane_num in roads
    tmp = df_memo[['RID', 'lane_num']].groupby('RID').agg( lambda x: stats.mode(x)[0][0] )
    tmp.rename(columns={"lane_num":'lane_num_new'}, inplace=True)

    DB_roads = DB_roads.set_index('RID')
    DB_roads = DB_roads.merge( tmp, left_index=True, right_index=True, how='left' )
    DB_roads.loc[:, 'lane_num'] = DB_roads.lane_num_new - 1
    DB_roads.drop(columns=['lane_num_new'], inplace=True)
    DB_roads.reset_index(inplace=True)
    
    # update the lane_num in panos
    tmp = DB_panos[['PID','DIR']].reset_index().merge(df_memo[['PID','DIR','lane_num']], on=['PID','DIR'])
    tmp.rename(columns={"lane_num":'lane_num_new'}, inplace=True)
    tmp.drop_duplicates(inplace=True)
    
    DB_panos = DB_panos.merge(tmp, on=['PID','DIR'], how='left')
    DB_panos.loc[:, 'lane_num'] = DB_panos.lane_num_new
    DB_panos.drop(columns=['lane_num_new', 'index'], inplace=True)
    DB_panos.describe()

    store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)
    
    return    


def gdf_to_postgis(gdf, name, engine=ENGINE, if_exists='replace', *args, **kwargs):
    try:
        gdf.to_postgis( name=name, con=engine, if_exists=if_exists )
        return True
    except:
        print('gdf_to_postgis error!')
    
    return False


def update_panos_url():
    """Update the url attribute of panos
    """
    pano_img_fns = pd.DataFrame( os.listdir(pano_dir), columns=['fn'])
    pano_img_fns.loc[:, 'PID'] = pano_img_fns.fn.apply(lambda x: x.split("_")[2] if len(x.split("_")) > 2 else None)
    pano_img_fns.drop_duplicates('PID', inplace=True)

    df_panos = load_DB_panos()
    indexes_with_imgs = df_panos.merge(pano_img_fns['PID'], on='PID').index
    df_panos.loc[indexes_with_imgs, 'url'] = df_panos.loc[indexes_with_imgs].PID.apply(lambda x: f"http://192.168.135.15:4000/pred_by_pid?format=img&pid={x}")
    
    gdf_to_postgis(df_panos, 'panos')

    return


#%%
if __name__ == '__main__':
    DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(new = False)

    # DB_panos.drop(columns='wgs', inplace=True)
    # DB_backup(DB_pano_base, DB_panos, DB_connectors, DB_roads)

    DB_pano_base.set_crs(epsg=4326)
    DB_pano_base.crs
    df_panos = gpd.read_postgis( "select * from panos", con=ENGINE, geom_col='geometry' )

    #! 存储去重问题
    DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(new = False)

    for att in ['Roads', 'Links']:
        DB_pano_base.loc[:, att] = DB_pano_base[att].apply(lambda x: eval(x))

    DB_pano_base.info()
    DB_pano_base.drop_duplicates()

    DB_panos.wgs.apply(lambda x: type(x))
    DB_pano_base.Links = DB_pano_base.Links.apply(lambda x: eval(x))

    DB_pano_base.Links.apply(lambda x: type(x)).unique()

    DB_panos.set_crs(epsg=4326, inplace=True)

    store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)
    
    
    extract_connectors_from_panos_respond( DB_pano_base, DB_roads )

    points = gpd.read_file('./points_longhua.geojson')
    
    points.merge(tmp, on=['PID','DIR'] ).describe()

    # update lane num in panos and roads
    # update_lane_num_in_DB()

