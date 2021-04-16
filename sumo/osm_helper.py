import os
from shapely.geometry import Point, LineString
import numpy as np
import geopandas as gpd


def osm_get(prefix, bbox   = '113.92348,22.57034,113.94372,22.5855'):
    # bounding box to in geo coordinates west,south,east,north
    prefix = "osm" if prefix is None else prefix
    cmd = f"python /usr/share/sumo/tools/osmGet.py -b {bbox} -p {prefix}"
    os.popen( cmd )
    
    return f"osm_{prefix}_bbox.osm.xml" if prefix != 'osm' else "osm_bbox.osm.xml"


def osm_parser(root):
    wayHash, nodeHash = {}, {}
    for element in root.findall('way'):
        id = int(element.get('id'))
        if id not in wayHash:
            wayHash[id] = {}
            wayHash[id]['elem'] = element
            wayHash[id]['points'] = [ int(i.get('ref')) for i in element.findall('nd')]
            
            for i in element.findall('tag'):
                wayHash[id][i.get('k')] = i.get('v')
            
    for node in root.findall('node'):
        id = int(node.get('id'))
        if id in nodeHash: continue

        info = {x.get('k'):x.get('v') for x in node.getchildren()} if node.getchildren() else {}
        info['pid'] = id
        info['xy'] = (float(node.get('lon')), float(node.get('lat')))
        info['geometry'] = Point( *info['xy'] )
        nodeHash[id] = info

    return wayHash, nodeHash


def add_coords_to_osm_node_hash(osm_nodeHash, OSM_CRS):
    """add projection coordinations of each node to osm node hash

    Args:
        osm_nodeHash (dict): osm node converted from osm xml file 

    Returns:
        [type]: [description]
    """
    assert OSM_CRS is not None, 'please process `parser_sumo_node_edge` to obtain `OSM_CRS`'
    df = gpd.GeoDataFrame(osm_nodeHash).T
    df = df.set_crs(epsg=4326).to_crs(epsg=OSM_CRS)
    df.loc[:, 'coords'] = df.geometry.apply(lambda x: [ round(i,2) for i in x.coords[0]])

    for i in osm_nodeHash.keys():
        osm_nodeHash[i]['coords'] = df.loc[i, 'coords']
        
    return osm_nodeHash


def tranfer_to_sumo(project_name='nanshan', osm_file="./osm_bbox.osm.xml"):
    res = os.popen(f" mkdir {project_name}; cp {osm_file} ./{project_name}/osm_bbox.osm.xml; \
                      cp ./start.sh ./{project_name} ; cd ./{project_name};sh start.sh").read()
    
    print(res)
    
    return True


def osm_edge_shape_to_linestring():
    tmp = edges.query( 'rid== 208128052 ' )

    tmp = gpd.GeoDataFrame(tmp, geometry=  tmp['shape'].apply( lambda x: LineString([ np.array(i.split(",")).astype(np.float)  for i in x.split(' ')])) )
    tmp.plot()

