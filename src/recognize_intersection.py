import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from shapely import geometry
from main import *
from coordTransfrom_shp import coord_transfer
from haversine import haversine_np

sys.path.append('/home/pcl/traffic/')
from map_factory import area_boundary

# .to_file( 'roads_baidu.geojson', driver="GeoJSON" )


def _short_link_combinations(edges, nodes_dic, save_path=None):
    """`Control short link combinations`, two-degree nodes (one incoming link and one outgoing link) will be removed, and two adjacent links will be combined to generate a new link.]

    Args:
        edges (gpd.GeodataFrame): edges
        nodes_dic (dict): node 
        save_json (bool, optional): Save the files or not. Defaults to True.

    Returns:
        [gpd.GeodataFrame]: The road network that combined the short link.
    """

    def split_roads_based_on_node_degree(edges, rid, nodes_dic):
        segment        = edges.query(f"road_id == '{rid}' ").sort_index()
        segment_points = segment.s.values.tolist() + [segment.e.values[-1]]
        
        split_ids      = [i for i, x in enumerate(segment_points) if x not in node_1_1]
        # [ split_ids.append(i) for i in [0, len(segment_points)-1] if i not in split_ids  ]
        for i in [0, len(segment_points)-1]:
            if i not in split_ids:
                split_ids.append(i)
        split_ids.sort()

        segments = []
        segments_id = []
        for i in range(len(split_ids)-1):
            segments.append(LineString( [nodes_dic[x][::-1] for x in segment_points[split_ids[i]: split_ids[i+1]+1]]))
            segments_id.append(segment_points[split_ids[i]: split_ids[i+1]+1])


        segments                = gpd.GeoDataFrame( {'geometry': segments, 'pids': segments_id}).reset_index()

        segments.loc[:, 'rid']  = rid
        segments.loc[:, 's']    = segments.pids.apply(lambda x: x[0])
        segments.loc[:, 'e']    = segments.pids.apply(lambda x: x[-1])
        segments.loc[:, 'pids'] = segments.apply(lambda x: ";".join(map(str, x.pids)), axis=1)

        def extract_att(att):
            tmp = segment[att].unique()[0] if len(segment[att].unique()) == 1 else segment[att].unique()
            segments.loc[:, att] = tmp

        for att in ['road_type', 'lanes', 'name', 'oneway', 'maxspeed']:
            extract_att( att )

        # segments.to_file('test.geojson', driver="GeoJSON")

        return segments

    out_degree, in_degree = edges.s.value_counts(), edges.e.value_counts()
    nodes_degree = pd.concat([out_degree, in_degree], axis=1).fillna(0).astype(np.int)
    node_1_1 = nodes_degree.query( "s==1 and e==1").index.astype(np.int).to_list()

    # shorten the roads
    rids, res = edges.road_id.unique(), []
    for rid in tqdm(rids, desc='Shorten the links'):
        res.append(split_roads_based_on_node_degree(edges, rid, nodes_dic))
    edges_new = pd.concat(res)
    edges_new = gpd.GeoDataFrame(edges_new, geometry=edges_new.geometry)

    if save_path is not None:
        edges_new.to_file(save_path, driver="GeoJSON")

    return edges_new

def _parse_node_list(nodelist, in_sys='wgs', out_sys='wgs'):
    node_dic = {}
    for node in tqdm(nodelist, desc='node parse'):
        node_id = node.getAttribute('id')
        node_lat = float(node.getAttribute('lat'))
        node_lon = float(node.getAttribute('lon'))
        node_dic[int(node_id)] = (node_lat, node_lon)

    node_dic = {}
    node_signal = []
    for node in tqdm(nodelist, desc="node traverse"):
        node_id = node.getAttribute('id')
        node_lat = float(node.getAttribute('lat'))
        node_lon = float(node.getAttribute('lon'))
        node_dic[int(node_id)] = (node_lat, node_lon)

        taglist = node.getElementsByTagName('tag')
        if taglist:
            res = {'id': int(node_id)}
            for tag in taglist:
                # res[tag.getAttribute('k')] = tag.getAttribute('v') 
                if tag.getAttribute('k') == 'traffic_signals':
                    res['traffic_signals'] = tag.getAttribute('v')

                if tag.getAttribute('k') == 'traffic_signals:direction':
                    res['direction'] = tag.getAttribute('v')

            if len(res)> 1:
                node_signal.append( res )

    nodes = pd.DataFrame(node_dic).T.rename(columns={0: 'y', 1: 'x'})
    nodes = gpd.GeoDataFrame(nodes, geometry=nodes.apply( lambda i: Point(i.x, i.y), axis=1))
    nodes = coord_transfer(nodes, in_sys, out_sys)
    nodes.loc[:, ['x']], nodes.loc[:, ['y']] = nodes.geometry.x, nodes.geometry.y

    node_signal = nodes.merge( pd.DataFrame( node_signal ), left_index=True, right_on='id' )
    # node_signal.to_file('node_signal.geojson', driver="GeoJSON")

    return node_dic, nodes, node_signal 

def _parse_way_list(waylist, nodes):
    edges = []

    for way in tqdm( waylist, desc='parse waylist: '):
        taglist = way.getElementsByTagName('tag')

        # if 'highway' in tag_key:
        #     for tag in taglist:
        #         if tag.getAttribute('k') == 'highway':
        #             road_flag = True
        #             road_type = tag.getAttribute('v')
        #             break
        # if road_flag:
        info = { tag.getAttribute('k'): tag.getAttribute('v') for tag in taglist }

        filter = ['cycleway', 
                  'corridor', # 人行天桥
                  'track', # 郊区、乡村、工矿区、田间、林间小路
                  'living_street', # 居住区车行道路，公园车行道路
                  'disused','footway','path','pedestrian','raceway', 'steps', 'service', 'services',
                  ]
        if 'highway' in info and info['highway'] not in filter :
            info['road_id'] = way.getAttribute('id')
            ndlist = way.getElementsByTagName('nd')
            nds = []
            for nd in ndlist:
                nd_id = nd.getAttribute('ref')
                nds.append(nd_id)

            # TODO 线段化
            for i in range(len(nds)-1):
                edges.append({'s': nds[i], 'e': nds[i+1], **info})

    edges = pd.DataFrame(edges)
    for att in ['s', 'e', 'road_id']:
        edges.loc[:, att] = edges.loc[:, att].astype(np.int)  

    # road_type_filter = ['motorway', 'motorway_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link']
    # edges.query(f'road_type in {road_type_filter}', inplace=True)

    # edges.to_file( 'shenzhen_edge.geojson', driver="GeoJSON" )
    return edges

def _post_parse_way_list(edges, nodes, nodes_lst:list=None, combine=True ):
    if nodes_lst is not None:
        edges.query( f"s in {nodes_lst} or e in {nodes_lst}", inplace=True )

    'drop useless columns'
    drop_atts = []
    for att in list(edges):
        if edges[att].nunique() > 1:
            continue
        drop_atts.append(att)
    print(f"Shrink Dataframe columns: {edges.shape[1]} -> {edges.shape[1] - len(drop_atts)}")
    
    edges.drop( columns=drop_atts, inplace=True )
    edges = edges.merge(nodes[['x', 'y']], left_on='s', right_index=True).rename(columns={'x': 'x0', 'y': 'y0'}
                ).merge(nodes[['x', 'y']], left_on='e', right_index=True).rename(columns={'x': 'x1', 'y': 'y1'})
    edges = gpd.GeoDataFrame(edges, geometry=edges.apply(lambda i: LineString([[i.x0, i.y0], [i.x1, i.y1]]), axis=1))
    edges.loc[:, 'l'] = edges.apply(lambda i: haversine_np( (i.y0, i.x0), (i.y1, i.x1))*1000, axis=1)

    atts = ['s',
            'e',
            'road_id',
            'highway',
            'lanes',
            'name',
            'oneway',
            'maxspeed' ,
            'motor_vehicle',
            'geometry']
    edges = edges[atts]
    edges.rename(columns={'highway':'road_type'}, inplace=True)
    edges.sort_values( by = ['road_id'], inplace=True )
    return edges

def get_road_network_from_osm(fn, mask='深圳',in_sys='wgs', out_sys='wgs'):
    # fn = '/home/pcl/Data/minio_server/input/shenzhen_road_osm.xml'
    import xml
    dom      = xml.dom.minidom.parse(fn)
    root     = dom.documentElement
    nodelist = root.getElementsByTagName('node')
    waylist  = root.getElementsByTagName('way')

    nodes_dic, nodes, node_signal = _parse_node_list(nodelist)
    edges = _parse_way_list(waylist, nodes)

    'clip the features'
    if mask is not None:
        # df, _     = area_boundary.get_boundary_city_level(mask)
        # area = area_boundary.get_boundary_district_level('深圳')
        area = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson')
        area = area.query( f"name =='南山区' " ).geometry.values[0]
        nodes     = gpd.clip( nodes, area )
        node_signal = gpd.clip( node_signal, area )

    nodes_lst = nodes.index.to_list()
    edges = _post_parse_way_list(edges, nodes, nodes_lst)

    edges_new = _short_link_combinations( edges, nodes_dic, "../output/edges_shenzhen_shorten.geojosn" )
    # node_signal.to_file("../output/signal_shenzhen.geojosn", driver="GeoJSON")
    # edges.to_file("../output/edges_shenzhen.geojosn", driver="GeoJSON")

    """ query the road name in Nanshan """
    # road_name_lst = list(edges.name.dropna().unique())
    # import pickle
    # pickle.dump(road_name_lst, open('road_name_lst_nanshan.pkl', 'wb'))

    """ road filter by road type """
    # road_type_filter = ['cycleway', 'disused','footway','path','pedestrian','raceway','residential', 'steps', 'service', 'services']
    # edges.query(f'highway not in {list(road_type_filter)}', inplace=True)

    # edges_new.query( "oneway == 'reversible'" )
    return nodes, edges_new

def roads_from_baidu_search_API(fn=os.path.join(input_dir, "road_memo.csv")):
    """ 从`百度地图`中获取路网 """
    df_roads = pd.read_csv(fn) if os.path.exists(fn) else pd.DataFrame(columns=['name'])
    search_respond = pd.concat(
        df_roads.respond.apply(lambda x: pd.DataFrame(eval(x)[0]['content'] if 'content' in eval(x)[0] else [])).values)
    # python – Pandas.dataframe.query() – 获取非空行(Pandas等效于SQL：“IS NOT NULL”)
    roads_respond = search_respond.query("road_id == road_id")
    # remove the missing values columns
    roads_respond.dropna(axis=1, how="all", inplace=True)
    roads_respond = roads_respond[~roads_respond.profile_geo.isnull()]
    roads_respond.drop_duplicates('name', keep='last', inplace=True)
    roads_respond.reset_index(drop=True, inplace=True)

    roads_respond.loc[:, 'class'] = roads_respond.cla.apply(lambda x: x[-1])
    roads_respond.loc[:, 'directions'] = roads_respond.profile_geo.apply( lambda x: float(x.split("|")[0]))

    # extract road segment
    def convert_to_lines(content):
        def points_to_line(line):
            return [ct.bd09_to_wgs84(*bd_mc_to_coord(float(line[i*2]), float(line[i*2+1]))) for i in range(len(line)//2)]

        directions, ports, lines = content.profile_geo.split('|')

        df = pd.DataFrame(lines.split(';')[:-1], columns=['coords'])
        # Six decimal places
        df = gpd.GeoDataFrame(df, geometry=df.apply(
            lambda x: LineString(points_to_line(x.coords.split(','))), axis=1))

        df.loc[:, 'name'] = content['name']
        df.loc[:, 'primary_uid'] = content['primary_uid']

        return df

    roads = pd.concat(roads_respond.apply(lambda x: convert_to_lines(x), axis=1).values)
    roads.loc[:, 's'] = roads.geometry.apply(lambda x: x.coords[0])
    roads.loc[:, 'e'] = roads.geometry.apply(lambda x: x.coords[-1])

    if True:
        # move useless attribut
        drop_atts = []
        for att in list(roads_respond):
            try:
                if roads_respond[att].nunique() == 1:
                    drop_atts.append(att)
            except:
                print(f"{att} unhashable type")

        if 'directions' in drop_atts:
            drop_atts.remove('directions')
        drop_atts += ['profile_geo']

        roads_respond.drop(columns=drop_atts, inplace=True)
        # df_roads_info.to_csv('df_roads_info.csv')

    return roads_respond, roads

def query_gdf_by_bbox(gdf, bbox=[113.929807, 22.573702, 113.937680, 22.578734]):
    # extract the roads of intrest
    from shapely.geometry import Polygon

    coords = [bbox[:2], [bbox[0], bbox[3]],
              bbox[2:], [bbox[2], bbox[1]], bbox[:2]]

    gdf.reset_index(drop=True)
    roi = gpd.clip(gdf, Polygon(coords)).index.tolist()
    roi = gdf.loc[roi]

    return roi

# if __name__ == '__main__':
fn = '/home/pcl/Data/minio_server/input/shenzhen_road_osm.xml'
roi = query_gdf_by_bbox( DB_roads, bbox=[113.929807, 22.573702, 113.937680, 22.578734])
map_visualize(roi, scale=0.05)
# map_visualize(roi.query(f"PID_end == PID_start"), color="red")





""" 从`百度地图`中挑取感兴趣的道路 """
# get_road_shp_by_search_API('乾丰二路')
df_roads_info, df_segments = roads_from_baidu_search_API()
# lst = ['打石一路', '创科路', '打石二路', "仙茶路", "兴科一街", "创研路", '石鼓路', '茶光路','乾丰一路','乾丰二路']
# df_segments.query(f"name in {lst}", inplace=True)
df_segments.drop(columns=['s','e'] ).to_file(  os.path.join(input_dir, 'shenzhen_road_baidu.geojson'), driver="GeoJSON" )
map_visualize(df_segments, scale=0.01)
df_roads_info.columns
edges = df_segments[['s', 'e']].values



nodes, edges_new = get_road_network_from_osm( '/home/pcl/Data/minio_server/input/shenzhen_road_osm.xml', '深圳' )





node_bak = nodes.copy()
edge_bak = edges_new.copy()
edges = edges_new

edges.loc[:, '_oneway'] = edges.oneway.fillna(False).replace( {'yes': True, '-1':True, 'reversible':True, 'no':False},  ).astype(np.bool)

two_way_edges = edges.query( '_oneway' )
one_way_edges = edges.query( 'not _oneway' )


two_way_edges.query( f"'link' in road_type" )

two_way_edges_bak = two_way_edges.copy()
two_way_edges = two_way_edges_bak.copy()

two_way_edges.loc[:, ['s','e']] = two_way_edges.loc[:, ['e','s']].values



class Node:
    """
    Define the node in the road network 
    """

    def __init__(self, id):
        self.val = id
        self.x, self.y = [float(i) for i in id.split(',')]
        self.prev = set()
        self.nxt = set()
        self.indegree = 0
        self.outdegree = 0

    def add(self, point):
        self.nxt.add(point)
        self.outdegree += 1

        point.prev.add(self)
        point.indegree += 1

    def check_0_out_more_2_in(self):
        return self.outdegree == 0 and self.indegree >= 2

    def move_nxt_to_prev(self, node):
        if node not in self.nxt:
            return False

        self.nxt.remove(node)
        self.prev.add(node)
        self.indegree += 1
        self.outdegree -= 1
        return True

    def move_prev_to_nxt(self, node):
        if node not in self.prev:
            return False

        self.prev.remove(node)
        self.nxt.add(node)
        self.indegree -= 1
        self.outdegree += 1
        return True

class Digraph:
    def __init__(self, v=0, edges=None, *args, **kwargs):
        import pandas as pd
        import geopandas as gpd
        from shapely.geometry import Point
        self.Vertex = v
        self.Edge = 0

        # key is node, value is neighbors
        self.graph = {}
        self.prev = {}
        if edges is not None:
            self.build_graph(edges)

        self.calculate_degree()

    def __str__(self):
        return ""

    def add_edge(self, start, end):
        for p in [start, end]:
            for g in [self.graph, self.prev]:
                if p in g:
                    continue
                g[p] = set()

        self.graph[start].add(end)
        self.prev[end].add(start)
        pass

    def build_graph(self, edges):
        for edge in edges:
            self.add_edge(*edge)
        return self.graph

    def calculate_degree(self,):
        df_degree = pd.merge(
            pd.DataFrame([[key, len(network.prev[key])]
                          for key in network.prev], columns=['coord', 'indegree']),
            pd.DataFrame([[key, len(network.graph[key])]
                          for key in network.graph], columns=['coord', 'outdegree']),
            on='coord'
        )

        df_degree = gpd.GeoDataFrame(df_degree,
                                     geometry=df_degree.coord.apply(
                                         lambda x: Point(*x))
                                     )

        self.degree = df_degree

        # df_degree.query( " indegree > 1 or outdegree >1 " )


# ! #TODO 信号灯交叉点识别












edges




network = Digraph(edges=df_segments[['s', 'e']].values)
network.graph, network.prev

df_degree = network.degree
# df_degree.drop('coord', axis=1).to_file( 'df_degree.geojson', driver="GeoJSON" )

map_visualize(df_degree.query(" indegree > 1 or outdegree >1 "))


out_degree, in_degree = edges.s.value_counts(), edges.e.value_counts()
nodes_degree = pd.concat([out_degree, in_degree], axis=1).fillna(0).astype(np.int)

node_1_1 = nodes_degree.query("s==1 and e==1").index.astype(np.int).to_list()

edges.info()

rids = edges.road_id.unique()

len(rids)


def split_roads_based_on_node_degree(edges, rid):
    segment = edges.query(f"road_id == '{rid}' ").sort_index()
    segment_points = segment.s.values.tolist() + [segment.e.values[-1]]

    split_ids = [i for i, x in enumerate(segment_points) if x not in node_1_1]

    for i in [0, len(segment_points)-1]:
        if i not in split_ids:
            split_ids.append(i)
    split_ids.sort()

    segments = []
    segments_id = []
    for i in range(len(split_ids)-1):
        segments.append(LineString(
            [node_dic[x][::-1] for x in segment_points[split_ids[i]: split_ids[i+1]+1]]))
        segments_id.append(segment_points[split_ids[i]: split_ids[i+1]+1])

    segments = gpd.GeoDataFrame(
        {'geometry': segments, 'pids': segments_id}).reset_index()
    segments.loc[:, 's'] = segments.pids.apply(lambda x: x[0])
    segments.loc[:, 'e'] = segments.pids.apply(lambda x: x[-1])

    segments.loc[:, 'rid'] = rid
    # segments.to_file('test.geojson', driver="GeoJSON")

    return segments
