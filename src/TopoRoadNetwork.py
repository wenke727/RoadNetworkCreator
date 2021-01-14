import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from haversine import haversine_np
import xml.dom.minidom
import sys, os
import yaml

sys.path.append('/home/pcl/traffic/map_factory')
from coordTransfrom_shp import coord_transfer

from utils.utils import load_config

config    = load_config()
pano_dir = config['data']['pano_dir']
input_dir = config['data']['input_dir']


def download_map(fn, bbox, verbose=False):
    if not os.path.exists(fn):
        if verbose:
            print("Downloading {}".format(fn))

        import requests
        url = f'http://overpass-api.de/api/map?bbox={bbox}'
        r = requests.get(url, stream=True)
        with open( fn, 'wb') as ofile:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    ofile.write(chunk)
    return True

def get_osm_map_by_city(city):
    import pinyin 
    from area_boundary import get_boundary_city_level

    osm_fn = os.path.join( input_dir, f"{pinyin.get(city, format='strip')}_road_osm.xml")
    area, _ = get_boundary_city_level(city, coords_sys_output='wgs')
    bbox = area.bounds.values[0]
    download_map(osm_fn, ','.join([str(x) for x in bbox]))
    return osm_fn


# class TopoRoad():
# TODO 构建类
def get_road_network(fn, fn_road, in_sys = 'wgs', out_sys = 'wgs' ):
    '@params: fn, fn_road'
    road_type_filter = ['motorway','motorway_link', 'primary', 'primary_link','secondary', 'secondary_link','tertiary']
    dom = xml.dom.minidom.parse(fn)
    root = dom.documentElement
    nodelist = root.getElementsByTagName('node')
    waylist  = root.getElementsByTagName('way')

    # nodes
    node_dic = {}
    for node in nodelist:
        node_id = node.getAttribute('id')
        node_lat = float(node.getAttribute('lat'))
        node_lon = float(node.getAttribute('lon'))
        node_dic[int(node_id)] = (node_lat, node_lon)
    
    nodes = pd.DataFrame(node_dic).T.rename(columns={0:'y', 1:'x'})
    nodes = gpd.GeoDataFrame( nodes, geometry= nodes.apply(lambda i: Point(i.x, i.y), axis=1) )
    nodes = coord_transfer(nodes, in_sys, out_sys)
    nodes.loc[:,['x']], nodes.loc[:,['y']]  = nodes.geometry.x, nodes.geometry.y

    # edges
    edges = []
    for way in waylist:
        taglist = way.getElementsByTagName('tag')
        road_flag = False
        road_type = None
        for tag in taglist:
            if tag.getAttribute('k') == 'highway':
                road_flag = True
                road_type = tag.getAttribute('v')
                break

        if road_flag:
            road_id = way.getAttribute('id')
            ndlist = way.getElementsByTagName('nd')
            nds,e = [], []
            for nd in ndlist:
                nd_id = nd.getAttribute('ref')
                nds.append( nd_id )
            
            # TODO 线段化
            for i in range( len(nds)-1 ):
                edges.append( [nds[i], nds[i+1], road_id, road_type] )

            # line = LineString([ node_dic[x][::-1] for x in map(int,  nds)])
            # edges.append( [road_id, road_type, line] )


    # df_edge = gpd.GeoDataFrame( edges, columns=['name', 'type', 'geometry'] )
    # df_edge.to_file( 'shenzhen_edge.geojson', driver="GeoJSON" )
    # df_edge.plot()


    edges = pd.DataFrame( edges ).rename(columns={0:'s', 1:'e',2: 'road_id', 3: 'road_type'})
    edges = edges.query( f'road_type in {road_type_filter}' )
    edges.loc[:, ['s','e']] = pd.concat((edges.s.astype(np.int64), edges.e.astype(np.int64)), axis=1)

    edges = edges.merge( nodes[['x','y']], left_on='s', right_index=True ).rename(columns={'x':'x0', 'y':'y0'}
                ).merge( nodes[['x','y']], left_on='e', right_index=True ).rename(columns={'x':'x1', 'y':'y1'})
    edges = gpd.GeoDataFrame( edges, geometry = edges.apply( lambda i: LineString( [[i.x0, i.y0], [i.x1, i.y1]] ), axis=1 ) )
    edges.loc[:,'l'] = edges.apply(lambda i: haversine_np((i.y0, i.x0), (i.y1, i.x1))*1000, axis=1)
    # edges.drop(columns=['x0','y0','x1','y1'], inplace=True)
    
    edges.to_file( 'shenzhen_edge.geojson', driver="GeoJSON" )
    # nodes filter
    ls = np.unique(np.hstack((edges.s.values, edges.e.values)))
    nodes = nodes.loc[ls,:]

    # fn_road = None
    if fn_road:
        road_speed = pd.read_excel(fn_road)[['road_type', 'v']]
        edges = edges.merge( road_speed, on ='road_type' )
    return nodes, edges.set_index('s')



if __name__ == "__main__":
    # fn = get_osm_map_by_city('深圳')
    fn = '/home/pcl/Data/minio_server/input/shenzhen_road_osm.xml'
    
    pass