from pano_base import *
import geopandas as gpd
import pickle

def main(name='龙华区'):  
    area = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson')
    area = area.query( f"name =='{name}'" )  

    from road_network import OSM_road_network
    osm_shenzhen = pickle.load(open("/home/pcl/traffic/data/input/road_network_osm_shenzhen.pkl", 'rb') )
    osm_shenzhen.edges

    roads = gpd.clip( osm_shenzhen.edges, area )
    
    
    for road_name in lst:
        try:
        traverse_panos_by_road_name(road_name, buffer=500, max_level=200)
    except:
        store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)
        e_lst.append(road_name)w