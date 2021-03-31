import os
import copy
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET
from shapely.geometry import Point, LineString

# Done

def osm_get(prefix, bbox):
    # bounding box to retrieve in geo coordinates west,south,east,north
    bbox   = '113.92348,22.57034,113.94372,22.5855'
    prefix = "osm" if prefix is None else prefix
    cmd = f"python /usr/share/sumo/tools/osmGet.py -b {bbox} -p {prefix}"
    os.popen( cmd )
    
    return f"osm_{prefix}_bbox.osm.xml" if prefix != 'osm' else "osm_bbox.osm.xml"


def osm_parser(root):
    wayHash = {}
    for element in root.findall('way'):
        id = element.get('id')
        if id not in wayHash:
            wayHash[id] = element
        
    nodeHash = {}
    for node in root.findall('node'):
        id = node.get('id')
        if id in nodeHash: continue
        
        nodeHash[id] = Point( float(node.get('lon')), float(node.get('lat')))

    return wayHash, nodeHash


def tranfer_to_sumo(project_name='nanshan', osm_file="./osm_bbox.osm.xml"):
    res = os.popen(f" mkdir {project_name}; cp {osm_file} ./{project_name}/osm_bbox.osm.xml; \
        cp ./start.sh ./{project_name} ; cd ./{project_name};sh start.sh").read()

    return True


def indent(elem, level=0):
    i = "\n" + level * "\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
            
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

    return



# developing
def update_lane(rid=208128052, val=4):
    return update_lane_of_element(wayHash[str(rid)], 'lanes', val)



def update_lane_of_element(way_element, key, val):
    tags = [tag.get('k') for tag in way_element.findall('tag')]


    # TODO 判断是否为双向道路，若是则需要将val/2 
    # <tag k="oneway" v="yes" />
    if key in tags:
        element = way_element.findall('tag')[tags.index(key)]
        element.set('v', str(val))

        return True

    lanes_info = ET.Element('tag', {'k': key, 'v': str(val)})
    way_element.append(lanes_info)

    return True


tree = ET.parse('./osm_bbox.osm.bak.xml')
root = tree.getroot()
wayHash, nodeHash = osm_parser(root)

rid = 231901941


def get_points( rid = 231901941, nodeHash=nodeHash ):
    pids = [ x.get('ref') for x in  wayHash[str(rid)].findall('nd')]
    df = gpd.GeoDataFrame({ 'id': pids, 'geometry': [ nodeHash[x] for x in pids]  })
    df.reset_index()
    
    return df, pids


def split_road_and_update_lanes_val(rid):
    # 实现路口拓宽功能
    # @parma
    splits=['6444510067', '7782982564']
    lane_values = [3,4,3]

    df, pids = get_points(rid)
    
    splits_index = [pids.index(i) for i in splits] + [len(pids)]
    intervals, prev = [], 0
    for i in splits_index:
        intervals.append( pids[prev: i+1] )
        prev = i

    for index, inter in enumerate(intervals):
        new_way = copy.deepcopy(wayHash[str(rid)])
        for node in new_way.findall('nd'):
            if node.get('ref') not in inter:
                new_way.remove( node )
        
        # TODO set a unique id 
        new_way.set('id', new_way.get('id') + str(index))
        update_lane_of_element(new_way, 'lanes', lane_values[index])
        
        root.append(new_way)

    root.remove( wayHash[str(rid)] )



split_road_and_update_lanes_val(rid)
root = indent(root)
tree.write('osm_bbox.osm.xml', encoding='utf-8')





update_lane(208128052, 4)
update_lane(529249851, 4)  # 高新中四道

update_lane(208128050, 2)  # 高新中四道
update_lane(208128051, 2)  # 高新中四道
update_lane(374508069, 1)

update_lane(231901941, 4)


tranfer_to_sumo()



# if __main__():
    # file = osm_get('lxd', bbox = '113.92348,22.57034,113.94372,22.5855')
    # tranfer_to_sumo( 'lxd', file )

#%%
# ! def add_revert_road()
rid = 208128052 # 科技中二路

tree = ET.parse('./osm_bbox.osm.bak.xml')
root = tree.getroot()
wayHash, nodeHash = osm_parser(root)

# origin
update_lane_of_element(wayHash[str(rid)], 'oneway', 'yes')
# revert edge
way =  copy.deepcopy(wayHash[str(rid)])
nodes = way.findall('nd')[::-1]

for node in nodes:
    way.remove(node)


df, pids = get_points(rid, nodeHash)

line_offset = LineString( [ nodeHash[p].coords[:][0] for p in pids] ).\
    parallel_offset(3/110/1000, 'Left', join_style=2, mitre_limit=5)

# gpd.GeoSeries([
#     LineString( [ nodeHash[p].coords[:][0] for p in pids] ),
#     line_offset
# ]).plot()

id = -int(pids[0])

for node in line_offset.coords[::-1]:
    node_element = ET.Element('node', { 'id': str(id),'lat': f"{node[1]}", 'lon':f"{node[0]}"})
    root.append(node_element)
    
    nd_elem = ET.Element('nd', { 'ref': str(id)})
    way.append( nd_elem )

    id -= 1



# update_lane_of_element(way, 'oneway', 'yes')
way.set('id', '-'+way.get('id') )
root.append(way)


root = indent(root)
tree.write('osm_bbox.osm.xml', encoding='utf-8')

tranfer_to_sumo()

# %%
# TODO 整体平移, 增加节点，

df, pids = get_points(rid, nodeHash)

line_offset = LineString( [ nodeHash[p].coords[:][0] for p in pids] ).\
    parallel_offset(20/110/1000, 'Left', join_style=2, mitre_limit=5)

# gpd.GeoSeries([
#     LineString( [ nodeHash[p].coords[:][0] for p in pids] ),
#     line_offset
# ]).plot()

start_id = -int(pids[0])

for node in line_offset.coords[::-1]
    node_element = ET.Element('node', { 'id': str(id),'lat': str(y), 'lon': str(x)})
    root.append(node_element)
    id -= 1
    
    nd_elem = ET.Element('nd', { 'ref': str(id))
    way.append( nd_elem )



# %%
