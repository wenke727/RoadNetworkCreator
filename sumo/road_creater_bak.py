
#%%
# 初始化

# rids_debug = [570468184, 633620767, 183402036, 107586308] # 深南大道及其辅道
# rids_debug = [ 208128058, 529249851, -529249851, 208128051,208128050, 208128048,489105647, 231901941,778460597] # -208128058
rids_debug = None # 深南大道及其辅道

SUMO_HOME = "/usr/share/sumo"
osm_file = './osm_bbox.osm.xml'
name = 'osm'
road_types_lst = ['trunk', 'primary', 'secondary']

file = open(f"../log/sumo-{datetime.datetime.now().strftime('%Y-%m-%d')}.log", 'w').close()

if matchingPanos is None:
    matchingPanos = MatchingPanos()
    # matchingPanos = MatchingPanos(None)
    # matchingPanos.add_lst(RID_set, df_edges, debug=True)
    # matchingPanos.save_memo()

osm_net = OSM_Net(file='./osm_bbox.osm.bak.xml', save_fn='./osm_bbox.osm.xml', logger=SUMO_LOG)

rids = _get_rid_by_road_type(road_types_lst)

matchingPanos.add_lst(rids if rids_debug is None else rids_debug, df_edges, debug=True)
OSM_MATCHING_MEMO = matchingPanos.memo

if rids_debug is None:
    osm_net.rough_tune(rids, OSM_MATCHING_MEMO, save=True)
else:
    osm_net.rough_tune(rids_debug if isinstance(rids_debug, list) else [rids_debug], OSM_MATCHING_MEMO, save=True)

assert _pre_process_fine_tune(name, osm_file, False), 'check `_pre_process_fine_tune` functions'

sumo_net = Sumo_Net('osm', logger=SUMO_LOG)
osm_net.add_sumo_net_node_to_osm(sumo_net)
osm_net.add_coords_to_node(OSM_CRS)

"""微调"""
if rids_debug is None:
    for rid in rids:
        try:
            modify_road_shape(rid, SUMO_LOG)
        except:
            print(f"rid: {rid} error!" )
            SUMO_LOG.error(f"rid: {rid} error!" )
            break
else:
    if isinstance(rids_debug, list):
        [ modify_road_shape(i, SUMO_LOG) for i in rids_debug ]
    else:
        modify_road_shape(rids_debug, SUMO_LOG)

assert _post_process_fine_tune(name, osm_file, False), 'check `_post_process_fine_tune` functions'


#%%


# process road matching:  59%|█████▊    | 72/123 [00:07<00:26,  1.94it/s] 572963459, no matching records after filter algorithm
# process road matching:  60%|██████    | 74/123 [00:14<01:37,  2.00s/it] 96327302, no matching records after filter algorithm
# process road matching:  65%|██████▌   | 80/123 [01:00<04:28,  6.26s/it] 488273278, no matching records after filter algorithm
# process road matching:  66%|██████▌   | 81/123 [01:05<04:07,  5.90s/it] 911272994, process error
# process road matching:  80%|███████▉  | 98/123 [02:21<01:37,  3.91s/it] 231787203, no matching records after filter algorithm
# process road matching:  82%|████████▏ | 101/123 [02:32<01:20,  3.65s/it] 533679822, no matching records after filter algorithm
# process road matching:  85%|████████▍ | 104/123 [02:45<01:11,  3.76s/it] 623050456, process error
# process road matching:  85%|████████▌ | 105/123 [02:47<00:55,  3.10s/it] 623050457, process error
# process road matching:  89%|████████▊ | 109/123 [03:06<00:58,  4.15s/it] 636236894, no matching records after filter algorithm
# process road matching:  91%|█████████ | 112/123 [03:19<00:43,  3.96s/it] 533894248, no matching records after filter algorithm
# process road matching:  92%|█████████▏| 113/123 [03:22<00:37,  3.75s/it] 833749485, process error

rid = 96327302
del OSM_MATCHING_MEMO[rid]
matchingPanos.add(rid)
get_and_filter_panos_by_osm_rid(rid, vis=True)

        
# %%
"""
    DONE:
    572963461 -> 匹配的elem_lst is [None] 
    107586308
    TODO
    911272994 # if panos is None or panos.shape[0] == 0: 
    25529503 # 2号道路，神奇失踪正在溯源
"""
rid = 572963460
sumo_net.plot_edge(rid)
sumo_net.get_edge_df_by_rid(rid)

osm_net.get_pids_by_rid(rid, sumo_net)

modify_road_shape(rid, SUMO_LOG)

# panos matching
panos = get_and_filter_panos_by_osm_rid( rid, vis=True, debug=False )




# %%
# ! 针对panos匹配的情况进行异常值处理，从图的连通性角度出发
panos

name_to_id = {'高新中四道': 529249851,
              '科技中二路': 208128052,
              '科苑北路': 231901939,
              '高新中二道': 208128050,
              '科技中三路': 208128048,
              '科技中一路': 278660698,
              '高新中一道': 778460597
              }

# %%

# FIXME rid = 107586308， 存在和 `220885829`合并的情况

# bug 线段延申的情况
# rid = 231405165
# modify_road_shape(rid, SUMO_LOG)

# bug 没有预测错误的项目
# rid = 636237018
# modify_road_shape(rid, SUMO_LOG)


# 45569111 生成错误, 因为线段被裁剪头部
# rid = 45569111
# modify_road_shape(rid, SUMO_LOG)

# 修改匹配算法，剪枝
rid = 107586308

#%%
#! 获取sumo_net的 拓扑 关系
rid = '107586308#7-AddedOnRampEdge'

df_edge = sumo_net.edge

index = df_edge.query(f"id=='{rid}' ").index[0]

road = df_edge.loc[index]
end_point = road['to'] 
rid = road['rid']

# %%
pre_ramps = df_edge[ (df_edge['to'] == road['from']) & (df_edge['rid'] != road['rid']) ]
pre_ramps_numLanes = pre_ramps.numLanes.astype(int).sum()

nxt_ramps = df_edge[ (df_edge['from'] == road['to']) & (df_edge['rid'] != road['rid']) ]
nxt_ramps_numLanes = nxt_ramps.numLanes.astype(int).sum()

lane_num = int(road['numLanes']) - max(pre_ramps_numLanes, nxt_ramps_numLanes)

sumo_net.update_edge_elem_lane_num(rid, lane_num)

# %%

# rid = '107586308#7'

def update_laneNum_for_AddRampEdge(rid, verbose=True):
    df_edge = sumo_net.edge

    index = df_edge.query(f"id=='{rid}' ").index[0]
    road = df_edge.loc[index]
    # rid = road['rid']

    if verbose:
        print("\n\n", road['id'], road['numLanes'])

    pre_ramps = df_edge[ (df_edge['to'] == road['from']) & (df_edge['rid'] != road['rid']) & (df_edge['type'] != road['type']) ]
    pre_ramps_numLanes = pre_ramps.numLanes.astype(int).sum()

    nxt_ramps = df_edge[ (df_edge['from'] == road['to']) & (df_edge['rid'] != road['rid']) & (df_edge['type'] != road['type']) ]
    nxt_ramps_numLanes = nxt_ramps.numLanes.astype(int).sum()

    tmp_lane = int(road['numLanes']) - max(pre_ramps_numLanes, nxt_ramps_numLanes)
    lane_num = tmp_lane if tmp_lane > 0 else int(road['numLanes'])

    status = sumo_net.update_edge_elem_lane_num(rid, lane_num)

    if status is False:
        print(f"update_laneNum_for_AddRampEdge {rid} failed")
    else:
        print(f"update_laneNum_for_AddRampEdge {rid} success")
    

    return 
    

# update_laneNum_for_AddRampEdge(rid='107586308#7-AddedOnRampEdge')    


# %%
add_ramps_edges = sumo_net.edge[sumo_net.edge['id'].str.contains('107586308') & sumo_net.edge['id'].str.contains('RampEdge')]['id'].values

# %%
for edge in add_ramps_edges:
    update_laneNum_for_AddRampEdge(edge)

assert _post_process_fine_tune(name, osm_file, False), 'check `_post_process_fine_tune` functions'


# %%
