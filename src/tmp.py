
# %%
# ! extract topo data
def _parse_road_and_links(pano):
    _roads = pano['Roads']
    
    # judge by `IsCurrent`
    cur_index = 0
    for idx, r in enumerate(_roads):
        if r['IsCurrent'] != 1:
            continue
        cur_index = idx
        break

    nodes = _roads[cur_index]['Panos']

    _rid, src, dst = _roads[cur_index]['ID'], nodes[0]['PID'], nodes[-1]['PID']
    res = [{'rid': _rid, 'src': src, 'dst': dst, 'link':False}]

    links = pano['Links']
    for link in links:
        info = {'src': pano.name, 'dst': link['PID'], 'link':True}
        res.append(info)
        
    return res


def get_topo_from_gdf_pano(gdf_panos, drop_irr_records=True):
    """[summary]

    Args:
        gdf_panos ([type]): [description]
        drop_irr_records (bool, optional): Remove irrelevant records (the origin or destiantion not in the key_panos). Defaults to True.

    Returns:
        [type]: [description]
    """

    topo = []
    for lst in tqdm(gdf_panos.apply(lambda x: _parse_road_and_links(x), axis=1).values):
        topo += lst

    df_topo = pd.DataFrame(topo)
    df_topo.drop_duplicates(df_topo.columns, inplace=True)

    if drop_irr_records:
        pano_ids = gdf_panos.index.values.tolist()
        df_topo.query("src in @pano_ids and dst in @pano_ids", inplace=True)

    # calculate the similarity 
    df_topo.loc[:,['dir_0', 'dir_1']] = df_topo.apply(lambda x: 
        {'dir_0': gdf_panos.loc[x.src]['MoveDir'], 'dir_1': gdf_panos.loc[x.dst]['MoveDir']}, 
        axis=1, result_type='expand')
    df_topo.loc[:, 'similarity'] = df_topo.apply(lambda x: math.cos( azimuth_diff(x.dir_0, x.dir_1) ), axis=1)
    df_topo.loc[~df_topo.link, 'similarity'] = 1

    df_topo.sort_values(['src', 'link', 'similarity'], ascending=[True, True, False], inplace=True)

    df_topo.set_index(['src', 'dst'], inplace=True)

    return df_topo


def azimuth_diff(a, b, unit='radian'):
    """calcaluate the angle diff between two azimuth

    Args:
        a ([type]): Unit: degree
        b ([type]): Unit: degree
        unit(string): radian or degree

    Returns:
        [type]: [description]
    """
    diff = abs(a-b)

    if diff > 180:
        diff = 360-diff

    return diff if unit =='degree' else diff*math.pi/180


pid = '09005700121709091037594139Y'
res = _parse_road_and_links(gdf_panos.loc[pid])

df_topo = get_topo_from_gdf_pano(gdf_panos)
df_topo

# df_roads = df_topo[~df_topo.link]
# df_links = df_topo[df_topo.link]

topo = df_topo.to_dict(orient='index')


# %%

id = '09005700121709091548023739Y'
df_topo.query( "src == @id or dst == @id" )

# %%
graph = {}

for src, dst, link in df_topo.reset_index()[['src', 'dst', 'link']].values:
    graph[src] = graph.get(src, set())
    graph[src].add(dst)

# for s, e, link in df_topo[['src', 'dst', 'link']].values:
#     graph[s] = graph.get(s, dict())
#     graph[s][e] = (link)

graph
#%%

def forward_bfs(node, max_layer=50):
    queue = deque([node])
    path = [node]

    visited_pid = set()
    visited_rid = []

    layer = 1
    while queue:
        for _ in range(len(queue)):
            cur = queue.popleft()
            visited_pid.add(cur)
            
            # for nxt in graph[cur]:
            for nxt, nxt_item in df_topo.loc[cur].iterrows():
                if nxt not in graph:
                    continue
                
                print(layer, (cur, nxt))
                
                if nxt in visited_pid:
                    continue
                
                if topo[(cur, nxt)]['similarity'] < 0.8:
                    break
                
                if not topo[(cur, nxt)]['link']:
                    visited_rid.append(topo[(cur, nxt)]['rid'])
                    if (nxt, nxt) in topo:
                        visited_rid.append(topo[(nxt, nxt)]['rid'])   
                
                path.append(nxt)
                queue.append(nxt)
                
                break
            
        layer += 1
        
        if layer > max_layer:
            break

    return visited_rid


# pid = '09005700121709091547432209Y'
# pid = '09005700121709091539399529Y'
pid = '09005700121709091036077249Y'
pid = '09005700121709031453099872S'
pid = '09005700122003211407076215O'
pid = '09005700122003211407319405O'

rids = forward_bfs(pid)

map_visualize(
    gdf_roads.loc[list(rids)]
)
# %%
query_df(df_topo, 'src', '09005700121709091548023739Y')
















# %%

# ! load `PCL` data
def load_data():
    gdf_pano = load_postgis('panos_lst')
    roads     = load_postgis('roads')

    roads.drop_duplicates('RID', inplace=True)
    roads.set_index('RID', inplace=True)

    attrs = ['Links', 'Roads']
    for i in attrs:
        gdf_pano.loc[:, i] = gdf_pano.apply(lambda x: eval(x[i]), axis=1)

    pd_links_len = gdf_pano.Links.apply(len)
    gdf_pano = gdf_pano[pd_links_len!=0]

    gdf_pano.drop_duplicates('ID', inplace=True)
    gdf_pano.set_index("ID", inplace=True)

    return gdf_pano, roads

# gdf_pano, roads = load_data()
# pano_dict = gdf_pano.to_dict(orient='index')

# %%
def get_pano_links(id, gdf_panos):
    links = gdf_panos.loc[id]['Links']
    
    return links
    
id = '09005700121709091035388809Y'
links = get_pano_links(id, gdf_panos)

queue = deque([])
for link in links:
    node = link['PID']
    if node in gdf_panos.index:
        print(node, 'exist')
        continue
    print(node)
    queue.append(node)


# %%

# TODO forward, backward, 提取topo
def bfs_forward_origin(rid, gdf_pano=gdf_panos, roads=gdf_roads, plot=True, degree_thres=15):
    queue = deque([roads.loc[rid]['PID_end']])
    visited = set()

    path = [rid]

    def _is_valid(cur, nxt):
        return abs(cur['MoveDir'] - nxt['MoveDir']) < degree_thres
    
    while queue:
        cur = queue.pop()
        if cur not in gdf_pano.index:
            continue
        
        cur_info = gdf_pano.loc[cur]

        if cur_info['RID'] in visited:
            continue
        
        for link in cur_info['Links']:
            nxt = link['PID']
            if nxt not in gdf_pano.index:
                print(f"nxt {nxt} not in gdf_pano")
                continue
            
            nxt_info = gdf_pano.loc[nxt]
            
            if not _is_valid(cur_info, nxt_info):
                print(f"{nxt_info['RID']} not valid")
                continue

            nxt_rid = gdf_pano.loc[nxt]['RID']
            nxt_rid_start, nxt_rid_end = roads.loc[nxt_rid][['PID_start','PID_end']]
            
            if link['PID'] != nxt_rid_start:
                print(f"{link['PID']} != {nxt_rid_start}, {link['RID']}")
                continue
            
            print(cur, nxt, cur_info['MoveDir'], nxt_info['MoveDir'], queue )
            queue.append(nxt_rid_end)
            path.append(nxt_info['RID'])

        visited.add(cur_info['RID'])

    if plot:
        roads.loc[path].plot()

    return path

# 创科路北行
rid = 'b1e1bb-bcf9-1d16-0e6d-c02c40'
# 创科路南行
rid = '4bb09e-46d2-40c0-6421-026285'
# 打石一路东行
rid = '81ce8c-d832-1db9-61dc-ee8b61'
# 打石一路西行
rid = 'd39bfb-b6e3-0ad8-582b-b97ccd'

bfs_forward_origin(rid)

