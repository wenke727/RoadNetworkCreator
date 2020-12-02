DB_roads.crs = "epsg:4326"
DB_roads.loc[:, 'length'] = DB_roads.to_crs('epsg:3395').length

# 使用频率很低的函数
def read_Roads(fn='./roads_guangming.geojson'):
    """
    read road from file
    """
    df_roads = gpd.read_file(fn)
    df_roads.query("name_1 == '光侨路' ", inplace=True)
    map_visualize(df_roads.query("name_1 == '光侨路' ").head(2))
    return df_roads

def getPort_for_segment(one_road):
    # obtain the port of each segment
    lst = list(one_road)
    lst.remove('geometry')
    ports = one_road[lst].merge(
        points[['geometry']], left_on='end', right_index=True)
    ports = gpd.GeoDataFrame(ports, geometry=ports.geometry)
    map_visualize(ports)

def recognize_main_line(df_roads: gpd.GeoDataFrame):
    """ # TODO 识别道路的主线

    Args:
        df_roads (gpd.GeoDataFrame): [description]

    Returns:
        [type]: [description]
    """

    return df_roads

def traverse_panos_in_road_old(road_one_way, df_pano=pd.DataFrame(), df_pano_all=gpd.GeoDataFrame()):
    length = road_one_way.shape[0]

    queue = deque([(0, road_one_way.loc[0, 'coords'])])
    while queue:
        index, node = queue.popleft()
        if index >= length - 1:
            continue

        # x, y = [float(x) for x in node.split(',')]
        x, y = node
        info, pano_record, _ = query_pano_detail_by_coord(x, y, df_pano, False)

        # if df_pano.query( f" crawl_coord == '{info.crawl_coord}' or pano_id == {info.pano_id} " ).shape[0]:
        #     nearest_road_id = index + 1
        # else:
        df_pano = df_pano.append(info, ignore_index=True)
        if pano_record is not None:
            df_pano_all = df_pano_all.append(pano_record, ignore_index=True)
            nxt_road_id = road_one_way.distance(
                pano_record.iloc[-1].geometry).nsmallest(1).index[0]

            while nxt_road_id <= index:
                nxt_road_id = min(nxt_road_id + 2, length - 1)
        else:
            dis = road_one_way.loc[index, 'dis_cum'] + 20
            ids = road_one_way.query(f"dis_cum > { dis }").index
            nxt_road_id = ids[0] if len(ids) > 0 else length - 1
            print('\tnxt_road_id, ', nxt_road_id)

        # map_visualize( pano_record )
        print(f'id {index} -> {nxt_road_id}, node: {node}')

        queue.append((nxt_road_id, road_one_way.loc[nxt_road_id, 'coords']))
        time.sleep(1)

    return df_pano, df_pano_all


#%%
# 遍历尝试
# step 1: 12679072.96, 2582262.52 -> 12679157.9, 2582278.94 ->
x, y = 12679072.96, 2582262.52
info, gdf_pano, status = query_pano(x, y)

map_visualize(gdf_pano)
links = pano_respond_parser(info)

if links is None:
    # 若没有links，则维持现状继续往前走
    x = (gdf_pano.iloc[-1].X * 2 - gdf_pano.iloc[-2].X)/100
    y = (gdf_pano.iloc[-1].Y * 2 - gdf_pano.iloc[-2].Y)/100
    print(f"\t no link->: {x}, {y}")

elif links.shape[0] == 1:
    port = gdf_pano.iloc[-1].geometry
    next_node_id = np.argmin(df_order_coords.distance(port))
    x, y = df_order_coords.loc[next_node_id].coords

    # x, y  = links.loc[0].X/100, links.loc[0].Y/100
    # # x = gdf_pano.iloc[-1].X/100 + (gdf_pano.iloc[-1].X - gdf_pano.iloc[-2].X)/100 * 1.5
    # # y = gdf_pano.iloc[-1].Y/100 + (gdf_pano.iloc[-1].Y - gdf_pano.iloc[-2].Y)/100 * 1.5
    dis = math.sqrt(pow(x-12679338.64, 2) + pow(y-2582321.06, 2))
    print(f"\t update: {x}, {y}: {dis}")

else:
    print("Link")

ax = map_visualize(DB_roads)
if links is not None:
    links.plot(ax=ax, color='blue')



#%%

PIDs = list(nxt_pano.PID.values)
# map_visualize(nxt_pano)


count = 0
while len(PIDs) > 0 and count < 5:
    nxt_PIDs = []
    for nxt in PIDs:
        if DB_panos.query( f" PID == '{nxt}' " ).shape[0] > 0:
            continue

        nxt = {'pano_id': nxt}
        info, df = query_IDs_By_panoID(nxt)
        print( nxt )
        # map_visualize( df )
        nxt_pano = parser_pano_respond( info )
        nxt_PIDs += list(nxt_pano.PID.values)
    
    PIDs = nxt_PIDs
    print('PIDs', PIDs)
    count += 1





# df_pano, df_pano_all = obtain_panos_info(df_order_coords[0:40].reset_index(), df_pano, df_pano_all)
df_pano, df_pano_all = obtain_panos_info(df_order_coords.reset_index(), df_pano, df_pano_all)


df_pano['Rname'].unique()

map_visualize(df_pano_all, 's')


df_pano_all.drop_duplicates()
df_pano_all.to_file('df_pano_all.geojson', driver="GeoJSON")
df_pano.to_hdf('df_nano.h5', key='pano')



df_pano_all = gpd.read_file('./df_pano_all.geojson')
ids = df_pano_all.drop_duplicates().index
df_pano_all[~df_pano_all.index.isin(ids)].root.unique()

df_pano_all[~df_pano_all.index.isin(ids)].plot()
df_pano.query( "pano_id == '01005700001312031243046415T' " )


######################
# 顺序排列的道路坐标点
df_order_coords.plot()

length = df_order_coords.shape[0]



length = 20

cur_id = 0
dis_thres = 20
cur_dis = 0

####
queue = deque( [0] )

while queue:
    cur_id = queue.popleft()
    if cur_id >= length - 1:
        break

    print(cur_id)
    x, y = df_order_coords.loc[cur_id, 'coords']
    info, gdf_pano, status = query_Pano_IDs_By_Coord(x, y, False)

    if not status:
        cur_dis += dis_thres
        nxt_id = df_order_coords.query( f"dis_cum > {cur_dis}" ).index[0]
        nxt_id = nxt_id if nxt_id < length-1 else length-1
    else:
        nxt_id = df_order_coords.distance( gdf_pano.iloc[-1].geometry ).nsmallest(1).index[0]
        if nxt_id < cur_id:
            nxt_id = cur_id + 1
        nxt_dis = df_order_coords.loc[nxt_id].dis_cum
        cur_dis = max(nxt_dis, cur_dis)
        
        parser_pano_respond(info)

    queue.append(nxt_id)



#%%




def obtain_panos_info(road_one_way):
    length = road_one_way.shape[0]
    begin = 2
    queue = deque([(begin, road_one_way.loc[begin, 'coords'])])
    while queue:
        index, node = queue.popleft()
        if index >= length - 1:
            continue    
            
        info, pano_record, status = query_Pano_IDs_By_Coord( *node, False )

        if status:
            links = parser_pano_respond(info)
            entrance_dir = info['MoveDir']

            links.DIR - entrance_dir

            nxt_road_id = road_one_way.distance( pano_record.iloc[-1].geometry ).nsmallest(1).index[0]
    
            while nxt_road_id <= index:
                nxt_road_id = min(nxt_road_id + 1, length - 1)
        else:
            dis = road_one_way.loc[index, 'dis_cum'] + 20
            ids = road_one_way.query( f"dis_cum > { dis }" ).index
            nxt_road_id = ids[0] if len(ids) > 0 else length - 1
            print( '\tnxt_road_id, ', nxt_road_id )

        # map_visualize( pano_record )
        print(f'id {index} -> {nxt_road_id}, node: {node}')

        queue.append( (nxt_road_id, road_one_way.loc[nxt_road_id, 'coords']) )
        time.sleep(1)
    
    return 
DB_pano_base, DB_panos, DB_connectors, DB_roads = pd.DataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()
obtain_panos_info( df_order_coords[:50] )



map_visualize(DB_roads)



# %%
