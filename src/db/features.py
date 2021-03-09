import os, sys
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
from shapely.geometry import Point

sys.path.append("..") 
from utils.utils import load_config
from utils.spatialAnalysis import create_polygon_by_bbox

config   = load_config()
pano_dir = config['data']['pano_dir']
ENGINE   = create_engine(config['data']['DB'])


def get_features(feature, bbox=None, in_sys='wgs84'):
    """creaet polygon by bbox(min_x, min_y, max_x, max_y)

    Args:
        feature: 'point' or 'line'
        bbox (list): [min_x, min_y, max_x, max_y]
        in_sys: the coordination system of bbox

    Returns:
        features
    """

    assert in_sys == 'wgs84', "the coordination must be wgs84"
    matching = {'point':'panos', 'line': 'roads' }
    
    if bbox is not None:
        tmp = create_polygon_by_bbox( bbox=bbox )
        sql = f"""
            select * from {matching[feature]} 
            where ST_Crosses( geometry, ST_GeomFromText('{tmp}', 4326) ) or 
                  ST_Within( geometry, ST_GeomFromText('{tmp}', 4326) )
            """
    else:
        sql = f"select * from {matching[feature]} "
    res = gpd.read_postgis( sql, geom_col='geometry', con=ENGINE )
    
    # res.to_file("./tmp.geojson", driver="GeoJSON")
    # with open("./tmp.geojson") as f:
    #     res = f.readlines()
        
    return res


if __name__ == '__main__':
    from utils.geo_plot_helper import map_visualize
    
    # df = gpd.GeoDataFrame([{ 'geometry': bbox, 'index':0 }])

    res = get_features(feature='line', bbox=[113.929807, 22.573702, 113.937680, 22.578734])

    map_visualize(res)