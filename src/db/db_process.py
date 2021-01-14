import os, sys
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
from shapely.geometry import Point

sys.path.append("..") 
from utils.utils import load_config


config   = load_config()
pano_dir = config['data']['pano_dir']
ENGINE   = create_engine(config['data']['DB'])


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

    DB_pano_base  = gpd.read_postgis( 'select * from pano_base', geom_col='geometry', con=ENGINE )
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
    
    # FIXME DB_connectors need to be re-constructed 
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


if __name__ == '__main__':
    DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(new = False)

    # DB_panos.drop(columns='wgs', inplace=True)
    DB_backup(DB_pano_base, DB_panos, DB_connectors, DB_roads)

    DB_pano_base.set_crs(epsg=4326)
    DB_pano_base.crs
    ##############
    df_panos = gpd.read_postgis( "select * from panos", con=ENGINE, geom_col='geometry' )

    from roadNetwork import map_visualize
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

