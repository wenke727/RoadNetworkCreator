import geopandas as gpd
import pandas as pd
from shapely import geometry
from sqlalchemy import create_engine
import os
from mapAPI import get_staticimage
from roadNetwork import map_visualize
import yaml
from utils.coord.coord_transfer import bd_mc_to_wgs_vector
from shapely.geometry import Point

with open(os.path.join( os.path.dirname(__file__), 'config.yaml')) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

pano_dir = config['data']['pano_dir']
ENGINE = create_engine(config['data']['DB'])


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
        # for df in [DB_pano_base, DB_panos, DB_connectors, DB_roads]: df.set_crs(epsg=4326,inplace=True)
        
        return DB_pano_base, DB_panos, DB_connectors, DB_roads

    DB_pano_base  = gpd.read_postgis( 'select * from pano_base', geom_col='geometry', con=ENGINE )
    for att in ['Roads', 'Links']:
        DB_pano_base.loc[:, att] = DB_pano_base[att].apply(lambda x: eval(x))

    DB_panos      = gpd.read_postgis( 'select * from panos', geom_col='geometry', con=ENGINE )
    DB_connectors = gpd.read_postgis( 'select * from connectors', geom_col='geometry', con=ENGINE )
    DB_roads      = gpd.read_postgis( 'select * from roads', geom_col='geometry', con=ENGINE )

    # name each dataframe
    DB_pano_base.name = 'pano_base'
    DB_panos.name = 'panos'
    DB_connectors.name = 'connectors'
    DB_roads.name = 'roads'
    
    return DB_pano_base, DB_panos, DB_connectors, DB_roads

def store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads):
    """
    store road data to DB
    """
    config_local = {"con": ENGINE, 'if_exists':'replace'}


    DB_pano_base_bak = DB_pano_base.copy()
    try:
        # for att in ['Roads', 'Links','ImgLayer', 'SwitchID', 'TimeLine']:
        #     DB_pano_base_bak.loc[:, att] = DB_pano_base_bak[att].apply(lambda x: str(x))
        #     # attrs_str = []
        for att in list(DB_pano_base):
            types =  DB_pano_base[att].apply(lambda x: type(x)).unique() if att !='geometry' else []
            if  list in types:
                # attrs_str.append(att)
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


def extract_connectors_from_panos_respond( DB_pano_base, DB_roads ):
    # FIXME DB_connectors need to re-construction 
    roads = DB_pano_base[(DB_pano_base.Links.apply( lambda x: len(x) > 0 )) &
                         (DB_pano_base.ID.isin(DB_roads.PID_end.unique().tolist()) )
                        ]

    def construct_helper(item):
        df =  pd.DataFrame(item.Links)
        df.loc[:, 'prev_pano_id'] = item.ID
        return df

    connectors = pd.concat(roads.apply( lambda x: construct_helper(x), axis=1 ).values.tolist())
    connectors = gpd.GeoDataFrame( connectors, 
                                   geometry = connectors.apply( lambda i: Point( bd_mc_to_wgs_vector(i)), axis=1 ),
                                   crs ='EPSG:4326'
                                )
    return connectors


def traverse_panos(df_panos):
    import time, random
    from tqdm import tqdm

    # 遍历照片
    RIDs = df_panos.RID.unique()
    for rid in  RIDs[:]:
        df = df_panos.query( f"RID == '{rid}' " )

        for index, item in df.iterrows():
            if not (item.Order == 0 or item.Order == df.shape[0]-2):
                continue
            res = get_staticimage( item.PID, item.DIR )
            if res is not None:
                time.sleep(random.uniform(2, 5))


def DB_backup(DB_pano_base, DB_panos, DB_connectors, DB_roads):
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


if __name__ == '__main__':
    DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(new = False)

    DB_panos.drop(columns='wgs', inplace=True)
    DB_backup(DB_pano_base, DB_panos, DB_connectors, DB_roads)

    DB_pano_base.set_crs(epsg=4326)
    DB_pano_base.crs
    ##############
    df_panos = gpd.read_postgis( "select * from panos", con=ENGINE, geom_col='geometry' )


    map_visualize(df_panos)


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

