import pandas as pd
import coordTransform_py.CoordTransform_utils as ct


df = pd.read_excel('./点位选择-匹配经纬度-1218.xlsx', '整合表')
df.dropna(inplace=True)

df.loc[:, 'bd09'] = df['经纬度'].apply( lambda x:  ct.gcj02_to_bd09( *map( float, x.split(','))))
df.loc[:, 'wgs84'] = df['经纬度'].apply( lambda x:  ct.gcj02_to_wgs84( *map( float, x.split(','))))

df.to_excel( './副本点位选择-匹配经纬度-1218.xlsx' )



import geopandas as gpd

df = gpd.read_file( "/home/pcl/Data/minio_server/input/roads_lxd_baidu.geojson" )
gdf_wgs_to_gcj(df)
df.to_file( "/home/pcl/Data/minio_server/input/roads_lxd_baidu_wgs.geojson", driver="GeoJSON" )

import sys
sys.path.append("/home/pcl/traffic/map_factory")
from coordTransfrom_shp import gdf_gcj_to_wgs, gdf_wgs_to_gcj

df_gcj = gdf_wgs_to_gcj(df)

df_gcj.to_file( "/home/pcl/Data/minio_server/input/roads_lxd_baidu_gcj.geojson", driver="GeoJSON" )
