import pandas as pd
import coordTransform_py.CoordTransform_utils as ct


df = pd.read_excel('./点位选择-匹配经纬度-1218.xlsx', '整合表')
df.dropna(inplace=True)

df.loc[:, 'bd09'] = df['经纬度'].apply( lambda x:  ct.gcj02_to_bd09( *map( float, x.split(','))))
df.loc[:, 'wgs84'] = df['经纬度'].apply( lambda x:  ct.gcj02_to_wgs84( *map( float, x.split(','))))

df.to_excel( './副本点位选择-匹配经纬度-1218.xlsx' )

