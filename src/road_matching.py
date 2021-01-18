from PIL.Image import Image
# from main import *
from utils.geo_plot_helper import map_visualize
import numpy as np
# from utils.utils import load_config
# config = load_config()


STEPS = 5

""" import road network from OSM """
import pickle
from recognize_intersection import OSM_road_network
osm_shenzhen = pickle.load(open("../input/road_network_osm_shenzhen.pkl", 'rb') )
df_nodes = osm_shenzhen.nodes
df_edges = osm_shenzhen.edges
df_edges.reset_index(drop=True, inplace=True)
df_edges.loc[:,'rid'] = df_edges.loc[:,'rid'].astype(np.int)
# df_edges.loc[:, 'geometry'] = df_edges.pids.apply( lambda x:  LineString([ osm_shenzhen.node_dic[float(i)] for i in  x.split(";")]) )
map_visualize(df_edges, scale=0.01)


""" visualize the distribution of the number of panos in each segements """
# road_segemnt_panos_num = pd.DataFrame(DB_panos.RID.value_counts()).rename( columns={"RID": "count"} )
# sns.kdeplot( road_segemnt_panos_num['count'] )


""" 匹配两个不同系统的道路网络，然后生成视频 """
#%%
from utils.spatialAnalysis import *

def points_dis_matrix(coords:np.array, unit="METER"):
    """caculate the distance of two coordinations in [lat, lon]

    Args:
        coords (np.array or LinString): [description]
        unit (str, optional): [description]. Defaults to "METER".

    Returns:
        [float]: The distance between the two coordination in METER
    """
    from haversine import haversine_np

    if isinstance(coords, LineString): coords = np.array(coords.coords)
    assert( isinstance(coords, np.ndarray) == True )

    dis_matrix = coords
    res = haversine_np( (dis_matrix[:,0],  dis_matrix[:,1]),  (dis_matrix[:, np.newaxis][:,:,0], dis_matrix[:, np.newaxis][:,:,1]) ) 
    if unit == "METER": res *= 1000
    
    return res

def cut_and_align_helper(base_line:pd.Series, compared_line:pd.Series):
    """cut and align the two input line

    Args:
        base_line (pd.Series): the sequence of coordination
        compared_line (pd.Series): the sequence of coordination
        vis_origin_data (bool, optional): [description]. Defaults to True.

    Returns:
        [LineString]: segements of compared_line
    """
    # base_line, compared_line = line2_, line1_ 
    # base_line, compared_line = line1_, line2_ 

    mask = compared_line.apply( lambda x: the_foot_point_on_line(x, base_line))
    # print(f'mask: {mask}')
    # 特殊情况，当一个线段包含了panos的一整条路段的时候
    if mask.sum() == 0:
        # 此时 base_line 是 panos途径， compared_line 为OSM途径
        road_segment = compared_line
        foot0 = get_foot_point( base_line.iloc[0], road_segment.iloc[0], road_segment.iloc[-1] )
        foot1 = get_foot_point( base_line.iloc[-1], road_segment.iloc[0], road_segment.iloc[-1] )

        l0 = {'x0': road_segment.iloc[0][0],
              'x1': road_segment.iloc[-1][0],
              'y0': road_segment.iloc[0][1],
              'y1': road_segment.iloc[-1][1],
            }

        l1 = {'x0': foot0[0],
              'x1': foot1[0],
              'y0': foot0[1],
              'y1': foot1[1],
            }

        coords_new = [foot0, foot1] if -30 <= angle_bet_two_line(l0, l1) <= 30 else [foot1, foot0]

        return LineString(coords_new)

    left, right = 0, len(mask)-1
    start, end = left, right
    while not mask.iloc[left]:
        left += 1 
    while not mask.iloc[right]:
        right -= 1 

    left_foot_point, right_foot_point  = [],[]
    if left != start:
        panos_segment = compared_line.iloc[left-1: left+1]
        if the_foot_point_on_line( base_line.iloc[0], panos_segment, ratio_thres=0 ):
            left_foot_point = get_foot_point( base_line.iloc[0], compared_line.iloc[left-1], compared_line.iloc[left] )
        if the_foot_point_on_line( base_line.iloc[-1], panos_segment, ratio_thres=0 ):
            left_foot_point = get_foot_point( base_line.iloc[-1], compared_line.iloc[left-1], compared_line.iloc[left] )
        # assert( len(left_foot_point)!=0 )

    if right != end:
        panos_segment = compared_line.iloc[right: right+2]
        if the_foot_point_on_line( base_line.iloc[0], panos_segment, ratio_thres=0 ):
            right_foot_point = get_foot_point( base_line.iloc[0], compared_line.iloc[right], compared_line.iloc[right+1] )
        if the_foot_point_on_line( base_line.iloc[-1], panos_segment, ratio_thres=0 ):
            right_foot_point = get_foot_point( base_line.iloc[-1], compared_line.iloc[right], compared_line.iloc[right+1] )
        # assert( len(right_foot_point)!=0 )

    # add the foot point to the new coords
    coords_new = compared_line.iloc[ left: right+1 ].values.tolist()
    if len(left_foot_point) > 0:
        coords_new = [left_foot_point] + coords_new
    if len(right_foot_point) > 0:
        coords_new = coords_new + [right_foot_point]

    return LineString( coords_new if len(coords_new) > 1 else coords_new *2 )

def cut_and_align(line1:LineString, line2:LineString, vis=False):
    """cut and align the lines, based on the foot pointer of each port on the other line.

    Args:
        line1 (LineString): line1
        line2 (LineString): [description]
        vis (bool, optional): Plot the lanes or not. Defaults to True.
    """
    # line1, line2 = item.geometry, road.geometry 
    assert( isinstance(line1, LineString) and isinstance(line2, LineString) )
    line1_ = pd.Series(line1.coords[:])
    line2_ = pd.Series(line2.coords[:])
    new_line1 = cut_and_align_helper( line1_, line2_ )
    new_line2 = cut_and_align_helper( line2_, line1_ )

    if vis:
        fig, ax = map_visualize( gpd.GeoSeries( [ LineString(line1_), LineString(line2_) ] ), color='gray', label='origin lane' )
        gpd.GeoSeries(new_line1).plot(ax=ax, label='new line1', color='red',  linestyle="-.")
        gpd.GeoSeries(new_line2).plot(ax=ax, label='new line2', color='blue', linestyle="-.")

        for line in [new_line1, new_line2]:
            gpd.GeoSeries([ Point(x) for x in line.coords[:]]).plot(ax=ax)

    return new_line1, new_line2

def line_interplation(line, vis=False):
    """线段按照等间距的方式插入点

    Args:
        line ([LineString]): [description]

    Returns:
        [type]: [description]
    """
    assert( isinstance(line, LineString) )

    coords     = np.array(line.coords)
    dis_mat    = points_dis_matrix( coords )
    coords_tmp = []

    np.max(dis_mat)
    for i in range( len(coords)-1 ):
        dis = dis_mat[i][i+1]
        n_sample = round(dis)-1 if round(dis)-1 > 1 else 1
        x = np.linspace( coords[i][0], coords[i+1][0], n_sample, False )
        y = np.linspace( coords[i][1], coords[i+1][1], n_sample, False )
        # y = np.interp( x, coords[i:i+2, 0], coords[i:i+2, 1])
        coords_tmp.append( np.vstack( [x, y] ).T)
    
    # add the last point of the segment
    coords_tmp.append( coords[-1][:] )

    coords_inter = np.vstack( coords_tmp )

    if vis:
        fig, ax = map_visualize( gpd.GeoSeries( [ Point(*x)  for x in coords_inter] ), scale=2 )
        gpd.GeoSeries([line]).plot(ax=ax)
    
    return LineString(coords_inter)

def angle_bet_two_linestring_ignore_inte_point(line1:LineString, line2:LineString ):
    """计算两条LineString的角度，仅仅考虑起点和终点

    Args:
        a (LineString or pd.Series): LineString 或者 pd.Series（包含的geometry为LineString）
        b (LineString): [description]

    Returns:
        [type]: 返回角度
    """
    a, b = line1.copy(), line2.copy()

    if isinstance(a, pd.Series): a = a.geometry
    if isinstance(b, pd.Series): b = b.geometry

    line_a, line_b = {}, {}
    line_a['x0'], line_a['y0'], line_a['x1'], line_a['y1'] = a.coords[0] + a.coords[-1]
    line_b['x0'], line_b['y0'], line_b['x1'], line_b['y1'] = b.coords[0] + b.coords[-1]

    return angle_bet_two_line(line_a, line_b)

def get_related_position(line1:LineString, line2:LineString):
    """Calculate the relative position of the starting point of the pano path

    Args:
        point (list): [description]
        road (LineString): [description]

    Returns:
        [type]: [description]
    """

    # line1 = road_candidates.geometry.iloc[0]
    # line2 = road.geometry

    coords = line2.coords[:]
    n = len(coords)

    def helper(point):
        pos = -1
        for i in range(n-1):
            pos = relation_bet_point_and_line(point, [*coords[i], *coords[i+1]] )
            # if i>= n-2: 
            #     return 2
            if 0 <= pos <= 1:
                pos = (i+pos)/(n-1)
                return pos

        return 0 if pos <0 else 1
    
    start = helper( line1.coords[0] )
    end   = helper( line1.coords[-1] )
    mid = (start + end) / 2

    return mid

def matching_panos_path_to_network( road, result, DB_roads=DB_roads, 
        vis=True, vis_step=False, save_fig=True, buffer_thres = 0.00005, angel_thres = 30):
    # find the matching path of panos for a special road based on the frechet distance

    # road = road_osm.iloc[9]
    # vis=True; vis_step=True; buffer_thres = 0.00005

    road_candidates = DB_roads[ DB_roads.intersects( road.geometry.buffer(buffer_thres) )].query( 'length > 0' )
    if road_candidates.shape[0] <=0:
        return None
    
    res_dis, res_ang = [], []
    for index, item in road_candidates.iterrows():
        if item.length == 0:
            res_dis.append(float('inf'))
            res_ang.append(90)
            continue
        
        # TODO: 若是两条线几乎垂直，可以考虑忽略了
        angel   = angle_bet_two_linestring_ignore_inte_point(item, road)
        res_ang.append(angel)

        
        if 90-angel_thres< angel < 90 + angel_thres:
            res_dis.append(float('inf'))    
        else:
            l0, l1  = cut_and_align( item.geometry, road.geometry )
            l0, l1  = line_interplation(l0), line_interplation(l1)
            dis, dp = frechet_distance_bet_polyline( l0, l1 )
            res_dis.append( dis *110*1000 )

            if not vis_step:
                continue

            fig, ax = map_visualize( gpd.GeoSeries( [ road.geometry ] ), color='black', label='OSM road' )
            gpd.GeoSeries( [ item.geometry ] ).plot(color='gray', label='Pano path', ax=ax )
            for line in [l0, l1]: gpd.GeoSeries([ Point(x) for x in line.coords[:]]).plot(ax=ax)
            plt.title( f"frechet dis: {dis*110*1000:.2f}" )
            plt.legend()

    # 汇总统计结果 
    road_candidates.loc[:, 'frechet_dis']    = res_dis
    road_candidates.loc[:, 'angel']          = res_ang
    road_candidates.loc[:, 'osm_road_id']    = road.rid
    road_candidates.loc[:, 'osm_road_index'] = road.name
    road_candidates.loc[:, 'related_pos']    = road_candidates.geometry.apply( lambda x: get_related_position(x, road.geometry) )
    road_candidates.sort_values(by='related_pos', inplace=True)
    
    result.append(road_candidates)
    rid = road_candidates.iloc[np.argmin(res_dis)].RID

    if vis:
        fig, ax = map_visualize( road_candidates, color='black', label='Pano paths', linestyle=':' )
        for index, item in road_candidates.iterrows():
            ax.text(*item.geometry.centroid.coords[0], 
                    f"{item.frechet_dis:.0f}, {item.angel:.0f},\n {item.related_pos:.2f}",
                    horizontalalignment='center', verticalalignment='center' 
                    )

        gpd.GeoSeries( [ road.geometry ] ).plot( color='red', label="OSM road", ax=ax )
        road_candidates.query( f"RID=='{rid}'" ).plot( color='blue', linestyle='--' , label = "match pano", ax=ax )
        plt.legend()
        if save_fig: plt.savefig( f'../log/match_process/{road.name}.jpg', pad_inches=0.1, bbox_inches='tight' )

    return rid


""" 划分路段的方向 -> 记录有rid，以及方向标记 """
# road_osm = df_edges.query( "name == '打石一路' " )
road_name = '打石一路'
road_osm = df_edges.query( f"name == '{road_name}' " )
roads_ids = road_osm.rid.unique()
road_id = roads_ids[0]
road_osm = road_osm.query( f"rid == {road_id}" )
map_visualize(road_osm)

road_osm.crs is None

road_osm.set_crs(epsg=4326, inplace=True)
road_osm.crs

linestring_length(road_osm)

result = []
for i in range(road_osm.shape[0]):
    matching_panos_path_to_network( road_osm.iloc[i], result, vis=True, vis_step=False )

matching = pd.concat(result)
seq =  matching.query( 'frechet_dis < 5' )
seq.drop_duplicates('RID', keep ='first', ignore_index=True).reset_index().to_file( f'../output/tmp_road_match_{road_name}_0.geojson', driver="GeoJSON" )



#%%
road_id = roads_ids[1]
road_osm = df_edges.query( f"name == '{road_name}' " )
road_osm = road_osm.query( f"rid == {road_id}" )
map_visualize(road_osm)


result = []
for i in range(road_osm.shape[0]):
    matching_panos_path_to_network( road_osm.iloc[i], result, vis=True, vis_step=False )

matching = pd.concat(result)
# TODO 其他数据较好的情况下，距离可以适当取大一点，或者匹配的道路数量仅有一条的时候，适量折减
seq =  matching.query( 'frechet_dis < 5' )
seq.drop_duplicates('RID', keep ='first', ignore_index=True).reset_index().to_file( f'../output/tmp_road_match_{road_name}_1.geojson', driver="GeoJSON" )


map_visualize(matching.query( 'frechet_dis < 5' ))




# matching.query( 'frechet_dis < 5' ).to_file( '../output/tmp_road_match.geojson', driver="GeoJSON" )

#%%









# #%%


# # TODO 获取某一道路所有的rid, 输出视频
# i = 0

# road = road_osm.iloc[i]
# matching_panos_path_to_network( road_osm.iloc[i], result,vis=True, vis_step=False )
# i+=1
# # line1_ = pd.Series(road.coords[:])
# # line2_ = pd.Series(road_candidates.iloc[2].coords[:])
# # cut_and_align_helper(road, road_candidates.iloc[2] )




# #%%

# #%%






# # TODO 在街景中添加位置示意图
# def plot_pano_and_its_view(pid, dir=None):
#     """绘制pano所在的路段，位置以及视角

#     Args:
#         pid ([type]): [description]
#     """
#     pid_record = get_pano_id_by_rid(rid).query( f"PID == '{pid}'" )
#     assert( len(pid_record) > 0 )
#     pid_record = pid_record.iloc[0]

#     if dir is None:
#         dir = pid_record.DIR
#     x, y = pid_record.geometry.coords[0]
    
#     fig, ax = map_visualize( DB_roads.query( f"RID == '{pid_record.RID}' " ), label="Lane" )

#     x0, x1 = ax.get_xlim()
#     aus_line_len = (x1-x0)/20
#     dy, dx = math.cos(dir/180*math.pi) * aus_line_len, math.sin(dir/180*math.pi) * aus_line_len
#     ax.annotate('', xy=(x+dx, y+dy), xytext= (x,y) ,arrowprops=dict(facecolor='blue', shrink=0.05, alpha=0.5))
#     gpd.GeoSeries( [Point(x, y)] ).plot(ax=ax, label='Pano' )

#     plt.axis('off')
#     plt.legend()
#     plt.tight_layout()
#     return fig


# position = plot_pano_and_its_view( pid = '09005700121709091541462499Y' )

# type(position)

# from PIL import Image
# img = Image.open('/home/pcl/traffic/RoadNetworkCreator_by_View/input/09005700011601091418266812P.jpg')

# plt.imshow(img.resize((1080*2, 720*2)))
# plt.imshow( img )

# # 将一张图粘贴到另一张图像上
# x, y = [ int(x/3) for x in img.size]
# location_illustration = img.resize((x, y))

# img.paste( location_illustration, [0,0,x,y] )


# def fig2data(fig):
#     """
#     fig = plt.figure()
#     image = fig2data(fig)
#     @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
#     @param fig a matplotlib figure
#     @return a numpy 3D array of RGBA values
#     """
#     import PIL.Image as Image
#     # draw the renderer
#     fig.canvas.draw()
 
#     # Get the RGBA buffer from the figure
#     w, h = fig.canvas.get_width_height()
#     buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
#     buf.shape = (w, h, 4)
 
#     # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#     buf = np.roll(buf, 3, axis=2)
#     image = Image.frombytes("RGBA", (w, h), buf.tostring())
#     image = np.asarray(image)
#     return image


# location_illustration = fig2data( plot_pano_and_its_view(pid = '09005700121709091541462499Y') )

# plt.imshow(location_illustration)


# plt.imshow(position)













# vis=True; buffer_thres = 0.0001
# area_bound = create_polygon_by_bbox(road.geometry.buffer(buffer_thres).bounds)
# road_candidates = DB_roads[ DB_roads.intersects( area_bound )]

# dis, dis_1 = [], []
# # TODO add fuction `cut and align` <- cut the paons road segements
# for index, item in road_candidates.iterrows():
#     print(item)
#     l1 = pd.Series(road.geometry.coords[:])
#     l2 = pd.Series(item.geometry.coords[:])

    

#     line1_cur, line2_cut = cut_and_align( road.geometry, item.geometry )

#     dis.append(hausdorff_bet_polyline( item, road ))
#     dis_1.append( frechet_distance_bet_polyline( item, road ) )

# # rid = road_candidates.iloc[np.argmin(dis)].RID
# rid = road_candidates.iloc[np.argmin(dis_1)].RID

# if vis:
#     fig, ax = map_visualize( road_candidates, lyrs='s', color='orange', label = "road candidates", scale=0.01 )
#     road_candidates.query( f"RID == '{rid}' " ).plot(ax=ax, color='red', label='matching road')
#     gpd.GeoSeries([road.geometry]).plot( ax=ax, color = 'blue', linestyle="-.", label='roads' )
#     plt.legend()


