from baidu_map import get_road_shp_by_search_API
from db.db_process import load_from_DB, store_to_DB, ENGINE
from utils.geo_plot_helper import map_visualize

from utils.spatialAnalysis import clip_gdf_by_bbox




DB_pano_base, DB_panos, DB_connectors, DB_roads = load_from_DB(new=False)


road, _, _ = get_road_shp_by_search_API( "打石一路" )

bbox = road.total_bounds


df_panos = clip_gdf_by_bbox( DB_panos, bbox )





points = set()
for coords in road.geometry.apply( lambda x: x.coords[:] ).values:
    for coord in coords:
        points.add( coord )

from shapely.geometry import Point
import pandas as pd
import geopandas as gpd

roads_point = gpd.GeoDataFrame([ Point(*i) for i in list(points) ], columns=['geometry'])


fig, ax = map_visualize( df_panos )
roads_point.plot(ax=ax)


# nxt: 重叠怎么计算呢





