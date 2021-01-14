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

def query_gdf_by_bbox(gdf, bbox=[113.929807, 22.573702, 113.937680, 22.578734]):
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


""" helper functions """
def distance_bet_point_line( point, line ):
    """caculate the distance from point to line, and return the position of foot point and distance
    1) the foot point is on the line, the value is in [0,1]; 
    2) the foot point is on the extension line of segment AB, near the starting point, the value < 0; 
    3) the foot point is on the extension line of segment AB, near the ending point, the value >1; 


    Args:
        point ([type]): [description]
        line ([type]): [description]

    Returns:
        [type]: [description]
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

def get_Foot_Point(point, line_p1, line_p2):
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

    return ( round(xn, 6), round(yn, 6))

def relation_bet_point_and_line( point, line ):
    """    Judge the realtion between point and the line, there are three situation:
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

def the_foot_point_on_line( point, line, ratio_thres=0.005 ):
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
    return  0-ratio_thres <= relation_bet_point_and_line(point, line_) <= 1+ratio_thres



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
    return res * 110 *1000 if unit =="METER" else res


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

    def get_coords(tmp):
        # the input could be: geometry or pd.Series with geometry
        if isinstance( tmp, pd.Series ):
            return [i for i in zip( tmp.geometry.coords.xy[0], tmp.geometry.coords.xy[1] )]
        return [i for i in zip( tmp.coords.xy[0], tmp.coords.xy[1] )]

    res, ca = frechet_distance( np.array(get_coords(u)), np.array(get_coords(v)) )
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
 

if __name__ == "__main__":
    # create_polygon_by_bbox( bbox=[113.929807, 22.573702, 113.937680, 22.578734] )

    P = np.array([[1,1], [2,1], [2,2]])
    Q = np.array([[2,2], [0,1], [2,4]])
    frechet_distance(P,Q)

    pass

# Q = np.array([[[2,2]], [[0,1]], [[2,4]]])

# Q.reshape(-1,2)
# np.linalg.norm( P- Q, keepdims=True )
# from haversine import haversine_np
# haversine_np( (P[:,0],  P[:,1]),  (Q[:, np.newaxis][:,:,0], Q[:, np.newaxis][:,:,1]) )

# (P[:,0],  P[:,1]),  (Q[:, np.newaxis][:,:,0], Q[:, np.newaxis][:,:,1])

# P - Q[:, np.newaxis]

