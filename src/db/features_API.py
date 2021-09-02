import os, sys
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
from shapely.geometry import Point, Polygon

sys.path.append("..") 
from utils.utils import load_config

config   = load_config()
pano_dir = config['data']['pano_dir']
ENGINE   = create_engine(config['data']['DB'])


def create_polygon_by_bbox(bbox):
    """creaet polygon by bbox(min_x, min_y, max_x, max_y)

    Args:
        bbox (list): [min_x, min_y, max_x, max_y]

    Returns:
        Polygon
    """

    coords = [bbox[:2], [bbox[0], bbox[3]],
              bbox[2:], [bbox[2], bbox[1]], bbox[:2]]
    
    return Polygon(coords)


def get_features(feature, bbox=None, geom=None, in_sys='wgs84'):
    """get road features in bbox(min_x, min_y, max_x, max_y)

    Args:
        feature: 'point' or 'line'
        bbox (list): [min_x, min_y, max_x, max_y]
        in_sys: the coordination system of bbox

    Returns:
        features
    """

    assert in_sys == 'wgs84', "the coordination must be wgs84"
    matching = {'point':'panos', 
                'line': 'roads', 
                'road': 'roads', 
                'edge': 'osm_edge_shenzhen', 
                'node': 'osm_node_shenzhen' }
    feature = feature if feature not in matching else matching[feature]
    # assert not (bbox is None and geom is None), "bbox and geom cann't both be 0"
    if geom is None and bbox is None:
        res = gpd.read_postgis( f"""SELECT * FROM {feature} """, 
                                geom_col='geometry', 
                                con=ENGINE )
        
        return res
    
    if geom is None:
        geom = create_polygon_by_bbox( bbox=bbox )
        
    sql = f"""SELECT * FROM {feature} 
              WHERE ST_Crosses( geometry, ST_GeomFromText('{geom}', 4326) ) or 
                    ST_Within( geometry, ST_GeomFromText('{geom}', 4326) )
            """
    res = gpd.read_postgis( sql, geom_col='geometry', con=ENGINE )
        
    return res


if __name__ == '__main__':
    from utils.geo_plot_helper import map_visualize
    from db.db_process import update_lane_num_in_DB
    # df = gpd.GeoDataFrame([{ 'geometry': bbox, 'index':0 }])

    import geopandas as gpd
    futian_area = gpd.read_file('../../cache/福田路网区域.geojson').iloc[0].geometry.wkt
    res = get_features(feature='line', geom=futian_area)
    res_road = get_features(feature='topo_osm_shenzhen_edge', geom=futian_area)

    res = get_features(feature='line', bbox=[113.929807, 22.573702, 113.937680, 22.578734])
    map_visualize(res)
    res.head(2).to_json()

    
    area = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson')
    area = area.query( "name =='龙华区'" )  
    tmp = area.iloc[0].geometry
    
    lines = get_features( 'line', geom=tmp )
    lines.lane_num.value_counts()
    
    
    points = get_features( 'point', geom=tmp )
    # lines.fillna(3, inplace=True)
    
    lines.to_file('./lines_longhua.geojson', driver="GeoJSON")
    points.to_file('./points_longhua.geojson', driver="GeoJSON")
