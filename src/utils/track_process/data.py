import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import coordTransform_py.CoordTransform_utils as ct

""" analysis the GPS data of Shenzhen taxis, and evaluate their performance when applied in lane detection in intersection  
    track_process 处理出租车GPS数据，评估是否可以用于车道识别（Lane Detection）

"""

df = pd.read_csv( "/home/pcl/Data/minio_server/gps/P_CZCGPS_20160101.csv", encoding='gb2312', header=None )
df.rename(columns={1: 'plate', 2:'time', 3:'x', 4:'y'}, inplace=True)

plate = df.plate.unique()[100]

track = df.query( f" plate =='{plate}' ")[['plate', 'time', 'x', 'y']]
ps = track.apply( lambda x: Point( * ct.gcj02_to_wgs84( x.x, x.y) ), axis=1 )
track = gpd.GeoDataFrame( track, geometry = ps )
track.plot()

track.info()


pd.to_datetime( track['time']  , format = '%Y%m%d %H:%M:%S') 


track.to_file('track.geojson', driver="GeoJSON")
