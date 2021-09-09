#%%
import pandas as pd 
import geopandas as gpd
from shapely.geometry import point
from db.db_process import save_to_geojson

from pano_base import pano_base_main
from DigraphOSM import load_net_helper
from pano_img import get_staticimage_batch
from matching import st_matching, load_trajectory
from panos_topo import combine_rids, get_panos_by_rids, get_trajectory_by_rid
from pano_predict import pred_trajectory, PRED_MEMO, update_unpredict_panos
from setting import CACHE_FOLDER, LXD_BBOX, SZU_BBOX, SZ_BBOX

from utils.geo_plot_helper import map_visualize
from utils.df_helper import load_df_memo, query_df
from db.db_process import save_to_geojson, save_to_db
from utils.douglasPeucker import dp_compress_for_points


#%%
df_pred_memo = load_df_memo(PRED_MEMO)

# step 1: download OSM data
net = load_net_helper(bbox=SZ_BBOX, combine_link=True, reverse_edge=True, overwrite=False, two_way_offeset=True)


# step 2: dowload pano topo
futian_area = gpd.read_file('../cache/福田路网区域.geojson').iloc[0].geometry
pano_base_res = pano_base_main(project_name='futian', geom=futian_area)
gdf_base  = pano_base_res['gdf_base']
gdf_roads = pano_base_res['gdf_roads']
gdf_panos = pano_base_res['gdf_panos']
# map_visualize( pano_base_res['gdf_roads'], scale=.01 )


# step 3: download pano imgs
pano_img_res = get_staticimage_batch(pano_base_res['gdf_panos'], 50, True)


# step 4: pano topo
traj_rid_lst, rid_uf = combine_rids(gdf_base, gdf_roads, plot=True)

#%%
# rid = '550a27-40c5-f0d3-5717-a1907d'
rid = '5a27e0-9221-0ef2-4074-4ff127'
# rid = 'cb7422-27d2-c73b-b682-a12ebd' # 深南大道辅道-市民中心段-东行
rid = 'edbf2d-e2f3-703f-4b9f-9d6819'

rids = get_trajectory_by_rid(rid, rid_uf, traj_rid_lst, gdf_roads, plot=True)
traj = get_panos_by_rids(rids, gdf_roads, gdf_panos, plot=False)

# step 5: predict trajectory
pred_res = pred_trajectory(traj, df_pred_memo)
pred_res['aerial_view']

#%%
# step 6: HMM
# FIXME
traj_comp = dp_compress_for_points(traj, 20, reset_index=True, verbose=False)
path = st_matching(traj_comp, net, plot=True)


#%%
# step 7: data fusing



# %%
# DEBUG 


map_visualize(traj_comp.iloc[:-1])
# %%
path = st_matching(traj_comp.iloc[:4], net, plot=True)

# %%

save_to_geojson(traj_comp, 'traj_debug.geojson')

# %%




# %%
# FIXME: 
[['727432-798c-f08f-16d4-851f95',
  '8e653d-ddba-f92f-819f-09f794',
  '0ac4cd-5d50-e300-ad2c-faa71d',
  '2fc65c-b993-fd1d-a9c6-2ba173',
  'f1b4d8-7494-fb88-054a-cc4a13',
  '514cba-89ea-b8d6-3de2-15f9ac',
  'd51f52-4ab6-cba6-dc4f-2fdf73',
  'edbf2d-e2f3-703f-4b9f-9d6819']]

get_trajectory_by_rid('586fa0-4623-8530-bcea-12e41d', rid_uf, traj_rid_lst, gdf_roads, plot=True)
