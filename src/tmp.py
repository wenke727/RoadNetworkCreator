from pano_base import *
from utils.spatialAnalysis import create_polygon_by_bbox, linestring_length

traverse_panos_by_road_name_new('环观南路')

traverse_panos_by_road_name_new('观乐路')

def check_pid_duplicate_in_folder( folder = '/home/pcl/Data/minio_server/panos_data/Futian/益田路' ):
    """Check whether there is any pid dulplicate in the folder
    """ 
    if not os.path.exists(folder): 
        os.mkdir(folder)
        return 
    
    lst = os.listdir(folder)
    df = pd.DataFrame(lst, columns=['fn'])
    df.loc[:, 'pid'] = df.fn.apply( lambda x: x.split("_")[-2] )

    df_count = pd.DataFrame(df.pid.value_counts())
    num = df_count.query('pid>1').shape[0]
    if num != 0:
        print( f"total num: {df.shape[0]}, dulpicate num: {df_count.query('pid>1').shape[0]}" )
    else:
        print( f"NO duplication: {folder}" )
    return


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
""" /home/pcl/traffic/RoadNetworkCreator_by_View/src/predict_lanes.py """

# useless
def get_heading_according_to_prev_road(rid):
    """ 可能会有多个值返回，如：
        rid = 'ba988a-7763-e9af-a5fb-dc8590'
    """
    # config   = load_config()
    # ENGINE   = create_engine(config['data']['DB'])
   
    sql = f"""SELECT panos.* FROM 
            (SELECT "RID", max("Order") as "Order" FROM
                (
                SELECT * FROM panos 
                WHERE "RID" in
                    (
                        SELECT "RID" FROM panos 
                        WHERE "PID" in 
                        (
                            SELECT prev_pano_id FROM connectors 
                            WHERE "RID" = '{rid}'
                        )
                    )
                ) a
            group by "RID") b,
            panos
        WHERE panos."RID" = b."RID" and panos."Order" = b."Order"
        """
    res = pd.read_sql( sql, con=ENGINE )
    res

    return res.DIR.values.tolist()


def calc_angle(item): 
    """计算GPS坐标点的方位角 Azimuth 

    Args:
        item ([type]): [description]

    Returns:
        [type]: [description]
    """
    angle=0
    
    x1, y1 = item.p0[0], item.p0[1]
    x2, y2 = item.p1[0], item.p1[1]
    
    dy= y2-y1
    dx= x2-x1
    if dx==0 and dy>0:
        angle = 0
    if dx==0 and dy<0:
        angle = 180
    if dy==0 and dx>0:
        angle = 90
    if dy==0 and dx<0:
        angle = 270
    if dx>0 and dy>0:
       angle = math.atan(dx/dy)*180/math.pi
    elif dx<0 and dy>0:
       angle = 360 + math.atan(dx/dy)*180/math.pi
    elif dx<0 and dy<0:
       angle = 180 + math.atan(dx/dy)*180/math.pi
    elif dx>0 and dy<0:
       angle = 180 + math.atan(dx/dy)*180/math.pi
    return angle


def calc_angle_for_df(df):
    coords = df.geometry.apply(lambda x: x.coords[0]) 
    df_new = pd.DataFrame()
    df_new['p0'], df_new['p1'] = coords, coords.shift(-1)

    df_new[:-1].apply(lambda x: calc_angle(x), axis=1)


def draw_network_lanes( fn = "../lxd_predict.csv", save_img=None ):
    """draw High-precision road network 

    Args:
        fn (str, optional): [description]. Defaults to "../lxd_predict.csv".
        save_img ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    from scipy import stats
    colors = ['black', 'blue', 'orange', 'yellow', 'red']
    widths = [0.75, 0.9, 1.2, 1.5, 2.5, 3]
    
    df = pd.read_csv(fn)
    df.loc[:, "RID"] = df.name.apply( lambda x: x.split('/')[-1].split('_')[-4] )
    df = df.groupby( 'RID' )[['lane_num']].agg( lambda x: stats.mode(x)[0][0] ).reset_index()
    df.loc[:, 'lane_num'] = df.lane_num - 1

    matching = DB_roads.merge( df, on = 'RID' )
    max_lane_num = matching.lane_num.max()

    fig, ax = map_visualize(matching, color='gray', scale=.05, figsize=(12, 12))
    for i in range(max_lane_num):
        matching.query(f'lane_num=={i+1}').plot(color = colors[i], linewidth = widths[i], label =f'{i+1} lanes', ax=ax)
    plt.legend()
    plt.close()
    
    if save_img is not None: plt.savefig(save_img, dpi =500)

    return matching