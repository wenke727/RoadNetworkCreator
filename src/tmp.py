from pano_base import *
from utils.spatialAnalysis import create_polygon_by_bbox, linestring_length

traverse_panos_by_road_name_new('环观南路')

traverse_panos_by_road_name_new('观乐路')



# store_to_DB(DB_pano_base, DB_panos, DB_connectors, DB_roads)



# area = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson')
# area = area.query( "name =='龙华区'" )  
# # area = area.query( "name =='南山区'" )  

# area.plot()


def get_unvisited_point(road_name = '民治大道', buffer=20):
    # TODO 识别没有抓取到数据的区域
    df_roads, ports, road_buffer = get_road_buffer(road_name, buffer)
    lst = []
    for x in df_roads.geometry.apply( lambda x: x.coords[:] ):
        lst += x

    points = gpd.GeoDataFrame( {'geometry':[ Point(i) for i in set(lst)]})
    points.loc[:, 'area'] = points.buffer(buffer/110/1000)
    points.reset_index(inplace=True)

    panos = get_features('point', points.total_bounds)
    points.set_geometry('area', inplace=True)

    visited = sorted(gpd.sjoin(left_df=points, right_df=panos, op='contains')['index'].unique().tolist())
    ans = points.query( f"index not in {visited} " )

    return ans 


#%%
BBOX = [113.92389,22.54080, 113.95558,22.55791] # 科技园中片区

# step 1: 读取龙华区的道路数据
df_edges = gpd.read_file('/home/pcl/Data/minio_server/input/edges_Shenzhen.geojson')
# area = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson')
# area = area.query( "name =='龙华区'" )  
area = create_polygon_by_bbox(BBOX)
tmp = gpd.clip(df_edges, area, True)

# step 2: 读取现有街景的数据
panos = get_features('point', geom=area)

# step 3: 作差
buffer = 3.75 *2
tmp.loc[:, 'area'] = tmp.buffer(buffer/110/1000)
tmp.set_geometry('area', inplace=True)
tmp.reset_index(inplace=True)

visited = sorted(gpd.sjoin(left_df=tmp, right_df=panos, op='contains')['index'].unique().tolist())
road_type_filter = ['residential']
df_unvisited = tmp.query( f"index not in {visited} and road_type not in {road_type_filter}" )

df_unvisited.plot()

#%%

# step 4: 开始采集新数据
#TODO 
unvisited_name = df_unvisited.name.dropna().unique()

from tqdm import tqdm 
for name in tqdm(unvisited_name[4:], desc=f'trasvers roads: '):
    print(name)
    try:
        traverse_panos_by_road_name_new(name)
    except:
        pass


# %%