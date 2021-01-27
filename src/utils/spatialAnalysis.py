from geopandas.geodataframe import GeoDataFrame
from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import math
import geopandas as gpd
from shapely.geometry import LineString

class Line_vector(object):
    def __init__(self, item):
        self.x1 = item['x0']
        self.y1 = item['y0']
        self.x2 = item['x1']
        self.y2 = item['y1']
        self.v = self.vector()
        self.l = self.length()

    def vector(self):
        c = (self.x1 - self.x2, self.y1 - self.y2)
        return c

    def length(self):
        d = math.sqrt(pow((self.x1 - self.x2), 2) + pow((self.y1 - self.y2), 2))
        return d


def angle_bet_two_line(a, b):
    '''
    @input: pandas.core.series.Series, pandas.core.series.Series
    @return: 角度
    '''
    a = Line_vector(a)
    b = Line_vector(b)
    return np.arccos( np.dot(a.v, b.v) / (a.l*b.l) )/math.pi*180


def create_polygon_by_bbox(bbox):
    """creaet polygon by bbox(min_x, min_y, max_x, max_y)

    Args:
        bbox (list): [min_x, min_y, max_x, max_y]

    Returns:
        Polygon
    """
    from shapely.geometry import Polygon

    coords = [bbox[:2], [bbox[0], bbox[3]],
              bbox[2:], [bbox[2], bbox[1]], bbox[:2]]
    
    return Polygon(coords)


def clip_gdf_by_bbox(gdf, bbox=[113.929807, 22.573702, 113.937680, 22.578734]):
    # extract the roads of intrest

    gdf.reset_index(drop=True)
    roi = gpd.clip(gdf, create_polygon_by_bbox(bbox)).index.tolist()
    roi = gdf.loc[roi]

    return roi


def cal_dis_matrix(df, xy_cols=['x','y']):
    """ caculate distance matrix of point set """
    from haversine import haversine_np
    # return kilometers
    dis_matrix = df[xy_cols].values
    return haversine_np( (dis_matrix[:,0],  dis_matrix[:,1]),  (dis_matrix[:, np.newaxis][:,:,0], dis_matrix[:, np.newaxis][:,:,1]) )


def linestring_length(df:gpd.GeoDataFrame, add_to_att=False):
    """caculate the length of LineString
    @return: pd:Series, length
    """
    # """" caculate the length of road segment  """
    # DB_roads.loc[:, 'length'] = DB_roads.to_crs('epsg:3395').length
    if df.crs is None:
        df.set_crs(epsg=4326, inplace=True)
    dis =  df.to_crs('epsg:3395').length
    
    if add_to_att:
        df.loc[:, 'length'] = dis
        return
    
    return dis
    

""" helper functions """
def distance_bet_point_line( point:list, line:list):
    """caculate the distance from point to line, and return the position of foot point and distance
    1) the foot point is on the line, the value is in [0,1]; 
    2) the foot point is on the extension line of segment AB, near the starting point, the value < 0; 
    3) the foot point is on the extension line of segment AB, near the ending point, the value >1; 


    Args:
        point (list): [x, y]
        line (list): [x0, y0, x1, y1]

    Returns:
        [float]: the distance to line
    """
    # print(line)
    pqx = line[2] - line[0]
    pqy = line[3] - line[1]
    dx  = point[0]- line[0]
    dy  = point[1]- line[1]
    # 线段长度的平方
    d = pow(pqx,2) + pow(pqy,2) 
    # 向量 点积 pq 向量（p相当于A点，q相当于B点，pt相当于P点）
    t = pqx*dx + pqy*dy

    flag = 1
    if(d>0): 
        t = t/d
        # flag： 起点 < 0 <= 线段中 <= 1 < 终点
        flag = t
    if(t<0): 
        t = 0
    elif(t>1): 
        t = 1
    dx = line[0] + t*pqx - point[0]
    dy = line[1] + t*pqy - point[1]
    return flag, math.sqrt(dx*dx + dy*dy)


def get_foot_point(point, line_p1, line_p2):
    """
    @point, line_p1, line_p2 : [x, y, z]
    """
    x0 = point[0]
    y0 = point[1]
    # z0 = point[2]

    x1 = line_p1[0]
    y1 = line_p1[1]
    # z1 = line_p1[2]

    x2 = line_p2[0]
    y2 = line_p2[1]
    # z2 = line_p2[2]

    # k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1) + (z1 - z0) * (z2 - z1)) / \
    #     ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)*1.0
    k = -((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2 )*1.0
    xn = k * (x2 - x1) + x1
    yn = k * (y2 - y1) + y1
    # zn = k * (z2 - z1) + z1

    return (round(xn, 6), round(yn, 6))


def relation_bet_point_and_line( point, line ):
    """Judge the realtion between point and the line, there are three situation:
    1) the foot point is on the line, the value is in [0,1]; 
    2) the foot point is on the extension line of segment AB, near the starting point, the value < 0; 
    3) the foot point is on the extension line of segment AB, near the ending point, the value >1; 

    Args:
        point ([double, double]): point corrdination
        line ([x0, y0, x1, y1]): line coordiantions

    Returns:
        [float]: the realtion between point and the line
    """
    pqx = line[2] - line[0]
    pqy = line[3] - line[1]
    dx  = point[0]- line[0]
    dy  = point[1]- line[1]
    # 线段长度的平方
    d = pow(pqx,2) + pow(pqy,2) 
    # 向量 点积 pq 向量（p相当于A点，q相当于B点，pt相当于P点）
    t = pqx*dx + pqy*dy

    flag = 1
    if(d>0): 
        t = t/d
        # flag： 起点 < 0 <= 线段中 <= 1 < 终点
        flag = t

    return flag


def the_foot_point_on_line( point:list, line:pd.Series, ratio_thres=0.000):
    """caculate the foot point is on the line or not

    Args:
        point (list): coordination (x, y)
        line (pd.Series): [description]
        ratio_thres (float, optional): [ratio threshold]. Defaults to 0.005.

    Returns:
        [bool]: locate on the lane or not
    """
    # if isinstance( line, pd.Series ):
    line_ = line.iloc[0] + line.iloc[-1]
    # line_ = line
    return  0 - ratio_thres <= relation_bet_point_and_line(point, line_) <= 1 + ratio_thres



""" hausdorff dist related fucntion """
def hausdorff(u: np.array, v: np.array):
    """cacaulte the hausdorff distance between u and v

    Args:
        u (ndarray): Input array
        v (ndarray): Input array

    Returns:
        double: The directed Hausdorff distance between arrays u and v,
    """
    from scipy.spatial.distance import directed_hausdorff
    # Ref: https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.spatial.distance.directed_hausdorff.html
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d


def hausdorff_bet_polyline( u: LineString or pd.Series, v: LineString or pd.Series, unit="METER" ):
    """cacaulte the hausdorff distance between u and v

    Args:
        u (LineString or pd.Serie): Input 1
        v (LineString or pd.Serie): Input 2

    Returns:
        double: The directed Hausdorff distance between arrays u and v
    """
    # Trajectory Similarity Measures, Ref: https://www.zhihu.com/question/27213170
    # TODO interpolation
    def get_coords(tmp):
        # the input could be: geometry or pd.Series with geometry
        if isinstance( tmp, pd.Series ):
            return [i for i in zip( tmp.geometry.coords.xy[0], tmp.geometry.coords.xy[1] )]
        return [i for i in zip( tmp.coords.xy[0], tmp.coords.xy[1] )]

    res = hausdorff( get_coords(u), get_coords(v) )
    return res * 110 *1000 if unit=="METER" else res


""" frechet dist related function """
def frechet_distance(P, Q):
    # Ref: [Frechet Distance距离算法详解](https://blog.csdn.net/weixin_42765516/article/details/104876099) 
    ca = np.ones((len(P), len(Q)))
    ca = np.multiply(ca,-1)
    res = frechet_dfs(ca, len(P) - 1, len(Q) - 1, P, Q, {})  # ca是a*b的矩阵(3*4),2,3
    return res, ca


def frechet_distance_bet_polyline( u: LineString or pd.Series, v: LineString or pd.Series, unit="METER", cut_or_not=True, interpolation=None ):
    """cacaulte the hausdorff distance between u and v

    Args:
        u (LineString or pd.Serie): Input 1
        v (LineString or pd.Serie): Input 2

    Returns:
        dis [double]: The directed Hausdorff distance between arrays u and v
        dp [matrix]: the matrix of dp
    """
    
    """
    大致解释下代码
    ca是一个矩阵，是n*m的矩阵，这里n和m是曲线1和2的集合长度，即曲线上点的个数。用于存放所有的计算结果。
    整体计算是这样：调用方法是传入的是两条曲线最后一个点的下标，然后递归调用，返回条件是一直从最后一个点的下标计算到【0】也就是第一个点。
    计算结果

    如果曲线P和曲线Q都是4个点，那么结果集就是P和Q从第一个点到最后一个点，每个点都计算其距离，其中最大的就是Frechet Distance距离
    如果曲线P是4个点，曲线Q是6个点，那么结果集是P和Q对应位置的点的距离以及P的第四个点和Q的第5和第6个点距离，其中最大的是Frechet Distance距离
    """
    # Trajectory Similarity Measures, Ref: https://www.zhihu.com/question/27213170

    def _get_coords(tmp):
        # the input could be: geometry or pd.Series with geometry
        if isinstance( tmp, pd.Series ):
            assert hasattr(tmp, "geometry"), "obj had no geometry attribiute"
            return [i for i in zip( tmp.geometry.coords.xy[0], tmp.geometry.coords.xy[1] )]
        return [i for i in zip( tmp.coords.xy[0], tmp.coords.xy[1] )]

    res, ca = frechet_distance( np.array(_get_coords(u)), np.array(_get_coords(v)) )
    res = res if unit =="METER" else res
    
    return res, ca 


def euc_dist(pt1, pt2, factor=1):
    # return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0]) + (pt2[1]-pt1[1])*(pt2[1]-pt1[1]))
    return np.linalg.norm(pt1 - pt2) * factor


def frechet_dfs(ca, i, j, P, Q, memo):
    if (i,j) in memo: return memo[(i,j)]
    # print(i, j)
    # print(ca, '\n')

    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = euc_dist(P[i], Q[j])
    elif i > 0 and j == 0:
        ca[i,j] = max(frechet_dfs(ca,i-1,0,P,Q, memo), euc_dist(P[i], Q[j]))
    elif i == 0 and j > 0:
        ca[i,j] = max(frechet_dfs(ca,0,j-1,P,Q, memo), euc_dist(P[i], Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(
            min(
                frechet_dfs(ca, i-1, j-1, P, Q, memo),
                frechet_dfs(ca, i-1, j,   P, Q, memo),
                frechet_dfs(ca, i,   j-1, P, Q, memo)
            ),
            euc_dist(P[i], Q[j])
            )
    else:
        ca[i,j] = float("inf")
    
    memo[(i,j)] = ca[i,j]
    return ca[i,j]
 

""" matching helper """
from shapely.geometry import Point, LineString
from .geo_plot_helper import map_visualize

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


if __name__ == "__main__":
    # create_polygon_by_bbox( bbox=[113.929807, 22.573702, 113.937680, 22.578734] )

    P = np.array([[1,1], [2,1], [2,2]])
    Q = np.array([[2,2], [0,1], [2,4]])
    frechet_distance(P,Q)

    pass

