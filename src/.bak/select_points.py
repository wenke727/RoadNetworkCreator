import geopandas as gpd

#! 筛选1000个点
df_edges.road_type.unique()
atts_filter = ['secondary', 'motorway', 'trunk']
tmp = df_edges.query( f"road_type in {atts_filter}" )

area = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson')
area = area.query( "name =='南山区'" )  

nanshan_edges = gpd.clip( tmp, area )

map_visualize( nanshan_edges  )

lst = nanshan_edges.pids.apply( lambda x: x.split(';') ).values

lst = [ pd.DataFrame(x) for x in lst ]

node_ids = pd.concat(lst, axis=0).rename(columns={0:'rid'}).sample(1050).rid.to_list()


nodes = df_nodes.reset_index().query( f"index in {node_ids}" )

map_visualize(nodes)



#%%

area = gpd.read_file('/home/pcl/Data/minio_server/input/Shenzhen_boundary_district_level_wgs.geojson')
area = area.query( "name =='龙华区'" )  







