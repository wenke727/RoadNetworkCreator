#%%
import geopandas as gpd

gdf = gpd.read_file("../cache/futian_topo_xucheng.geojson")

gdf
#%%


gdf.loc[:, "lane_order"] = gdf.laneid.apply(lambda x: int(x.split('_')[-1]))

gdf.sort_values(['segmentid','lane_order'], ascending=[True, False]).groupby('segmentid').head(1).plot()



# %%

gdf.sort_values()('lane', ascending=)
