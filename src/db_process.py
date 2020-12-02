import geopandas as gpd
from sqlalchemy import create_engine
import os
from mapAPI import get_staticimage
from roadNetwork import map_visualize
import yaml

config = yaml.load( open('./config.yaml') )
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
            gpd.GeoDataFrame(), gpd.GeoDataFrame(),gpd.GeoDataFrame(),gpd.GeoDataFrame()
        for df in [DB_pano_base, DB_panos, DB_connectors, DB_roads]: df.crs = "epsg:4326"
        
        return DB_pano_base, DB_panos, DB_connectors, DB_roads

    DB_panos      = gpd.read_postgis( 'select * from panos', geom_col='geometry', con=ENGINE )
    DB_pano_base  = gpd.read_postgis( 'select * from pano_base', geom_col='geometry', con=ENGINE )
    DB_connectors = gpd.read_postgis( 'select * from connectors', geom_col='geometry', con=ENGINE )
    DB_roads      = gpd.read_postgis( 'select * from roads', geom_col='geometry', con=ENGINE )

    return DB_pano_base, DB_panos, DB_connectors, DB_roads


def store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads):
    """
    store road data to DB
    """
    config_local = {"con": ENGINE, 'if_exists':'replace'}

    try:
        DB_pano_base.to_postgis( name='pano_base', **config_local )
        DB_panos.to_postgis( name='panos', **config_local )
        DB_connectors.to_postgis( name='connectors', **config_local )
        DB_roads.to_postgis( name='roads', **config_local)

        return True
    except:
        print('Store_to_DB failed!')
        return False


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

if __name__ == '__main__':
    DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(new = False)


    DB_pano_base = gpd.GeoDataFrame(DB_pano_base,
                                geometry = DB_pano_base.apply( lambda x:  Point(bd_mc_to_wgs(*[float(i) for i in x.res_coord.split(',')],1 )), axis=1 )
                )

    df_panos = gpd.read_file('../output/DB_panos.geojson')
    df_panos.to_postgis( name='panos', con=ENGINE, if_exists='replace' )

    PANO_FOLDER = "../output/panos"

    get_staticimage( '09005700121902131650367449U', 76 )
    get_staticimage( '09005700121902131650481199U', 77 )

    ##############
    df_panos = gpd.read_postgis( "select * from panos", con=ENGINE, geom_col='geometry' )


    map_visualize(df_panos)
