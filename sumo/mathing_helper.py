#%%
import os, sys
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas.api.types import CategoricalDtype
from shapely.geometry import LineString
import matplotlib.pyplot as plt

sys.path.append('../src')
from utils.geo_plot_helper import map_visualize
from road_matching import get_panos_of_road_and_indentify_lane_type_by_id, df_edges, DB_panos

#%%

# FIXME Fix the matching record problem, filter some unrelated record.
# matching related
class MatchingPanos():
    
    def __init__(self, df_edges, cache_folder="../cache",*args):
        self.memo = {}
        self.memo_df = None
        self.cache_folder = cache_folder
        self.error_roads_lst = []
        self.df_edges = df_edges
        if self.cache_folder is not None:
            self.load_memo()    

    
    def load_memo(self):
        if not os.path.exists(f'{self.cache_folder}/matching_panos_memo.pkl'):
            print(f"MatchingPanos loading {len(self.memo)} road from memo.pkl failed!")
            return False
        
        self.memo = pickle.load( open(f'{self.cache_folder}/matching_panos_memo.pkl', 'rb') )
        print(f"MatchingPanos loading {len(self.memo)} road from memo.pkl success!")
        return True


    def save_memo(self):
        pickle.dump(self.memo, open(f'{self.cache_folder}/matching_panos_memo.pkl', 'wb'))
        return True


    def convert_memo_to_df(self):
        """Convert the memo record to dataframe

        Returns:
            [type]: [description]
        """
        return pd.DataFrame(self.memo).T


    def matching_lst(self, lst, vis=False, debug=False):
        """Matching `rids` with OSM edges file.

        Args:
            lst (list): The list that need to be matching.
            df_edges ([type]): [description]
            vis (bool, optional): [description]. Defaults to False.
            debug (bool, optional): [description]. save imgs to folder `.cache`.
        """
        for i in tqdm(lst, 'process road matching: '):
            self.matching(i, vis, debug)

        return

    
    def matching(self, rid, vis=False, debug=False):
        """Matching `rid` with OSM edges file.

        Args:
            i ([type]): [description]
            vis (bool, optional): [description]. Defaults to False.
            debug (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        i = rid
        if i in self.memo:
            return
        
        self.memo[i] = self.memo.get(i, {})
        df, _ = get_and_filter_panos_by_osm_rid(i, self.df_edges, vis=vis, debug=debug, outlier_filter=True, verbose=False)
        if df is None:
            self.error_roads_lst.append(i)
            self.memo[i]['df'] = None
            self.memo[i]['median'] = None
            self.memo[i]['mode'] = None
            
            return False
        
        self.memo[i]['df'] = df
        self.memo[i]['median'] = int(df.lane_num.median())
        self.memo[i]['mode'] = int(df.lane_num.mode()[0])
        
        return True

    
    def save_rid_matching_df_to_file(self, rid, folder=None):
        """save the mathching record in the file

        Args:
            rid ([type]): [description]
            folder (dir): the path to store the dataframe if not `None` 

        Returns:
            [type]: [description]
        """
        if rid not in self.memo:
            print('please check the rid in the road set or not')
            return None
        
        df = self.memo[rid]['df']
        df.loc[:, 'RID'] = df.loc[:, 'RID'].astype(str)
        df.reset_index(inplace=True)
        
        if folder is not None:
            df.to_file( os.path.join(folder, f'{rid}.geojson'), driver="GeoJSON")
        
        return df


    def plot_matching(self, rid, *args, **kwargs):
        """plot the matching panos and show its lanenum in a color theme map

        Args:
            rid ([type]): [description]

        Returns:
            [type]: [description]
        """
        df = self.save_rid_matching_df_to_file(rid)
        if df is None:
            print(f"plot rid matching error, for the geodataframe {rid} is None")
        
        df.loc[:, 'lane_num'] = df.loc[:, 'lane_num'].astype(str)
        _, ax = map_visualize(df, color='gray', scale=0.05, *args, **kwargs)
        df.plot(column='lane_num', ax=ax, legend=True)
        
        return df
    

    @property
    def size(self):
        return len(self.memo)


def _get_revert_df_edges(road_id, df_edges, vis=False):
    """create the revert direction edge of rid in OSM file

    Args:
        road_id ([type]): the id of road
        df_edges ([type]): gdf create by 
        vis (bool, optional): plot the process or not. Defaults to False.

    Returns:
        [gdf]: the geodataframe of revert edge
    """
    road_id = road_id if road_id > 0 else -road_id
    df_tmp = df_edges.query(f"rid == {road_id} ")

    df_tmp.rid = -df_tmp.rid
    df_tmp.loc[:, ['s','e']] = df_tmp.loc[:, ['e','s']].values
    df_tmp.loc[:, 'index'] = df_tmp['index'].max() - df_tmp.loc[:, 'index']
    df_tmp.loc[:, 'geometry'] = df_tmp.geometry.apply( lambda x: LineString(x.coords[::-1]) )
    df_tmp.loc[:, 'pids'] = df_tmp.pids.apply( lambda x: ";".join( x.split(';')[::-1] ) )
    df_tmp.sort_values(by='index', inplace=True)
    # gpd.GeoDataFrame(pd.concat( [df_edges.query(f"rid == {road_id} "), df_tmp] )).to_file('./test.geojson', driver="GeoJSON")

    if vis:
        matching0 = get_panos_of_road_and_indentify_lane_type_by_id(-road_id, df_tmp, False)
        matching1 = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_edges, False)
        _, ax = map_visualize(matching0, scale =0.001)
        matching1.plot(column='level_0', legend=True, ax=ax, cmap='jet')
        matching0.plot(column='level_0', legend=True, ax=ax, cmap='jet')

    return df_tmp


def _panos_filter(panos, trim_nums=1):
    """Filter panos by:
    1. trim port
    1. lane_detection continues
    1. abs(lane_num - @median) < 2

    Args:
        panos (df): The origin df.
        trim_nums (int, optional): the trim length of the begining adn end points. Defaults to 1.

    Returns:
        [pd.df]: the filtered panos
    """
    if panos.shape[0] == 2 and panos.lane_num.nunique() == 1:
        return panos

    median = int(np.median(panos.lane_num))
    remain_ponas_index = np.sort(panos.Order.unique())[trim_nums: -trim_nums]

    tmp = panos[['Order','lane_num']]
    prev = panos.lane_num.shift(-1) == panos.lane_num
    nxt = panos.lane_num.shift(1) == panos.lane_num
    not_continuous = tmp[(prev|nxt) == False].Order.values.tolist()
    
    panos.query( f" Order not in {not_continuous} \
                    and Order in @remain_ponas_index \
                    and abs(lane_num - @median) < 2", 
                    inplace=True 
                )
    
    return panos


def get_and_filter_panos_by_osm_rid(road_id, df_edges, offset=1, vis=False, debug=False, outlier_filter=True, mul_factor=2, verbose=False):
    """Get the panos by OSM rid, and then filtered by some conditions.

    Args:
        road_id (int, optional): [description]. Defaults to 243387686.
        vis (bool, optional): [description]. Defaults to False.
        offset (int, optional): [the attribute `lane_num` is the real lane num or the real lane line num. If `lane_num` represent line num, then offset is 1. Other vise, the offset is 0 ]. Defaults to 1.

    Returns:
        matchingPano [dataframe]: [description]
        fig [plt.figure]: Figure
    """
    # step 1: matching panos
    atts = ['index', 'RID', 'Name', 'geometry', 'lane_num', 'frechet_dis', 'angel', 'osm_road_id', 'osm_road_index', 'related_pos', 'link']
    try:
        if road_id > 0:
            matching = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_edges, False) 
            matching = matching[atts].merge(df_edges[['s', 'e']], left_on='osm_road_index', right_index=True)
            road_name = df_edges.query(f'rid=={road_id}').name.unique()[0]
        else:
            # FIXME -208128058 高新中三道, 街景仅遍历了一遍
            df_tmp = _get_revert_df_edges(road_id, df_edges)
            road_name = df_tmp.name.unique()[0]
            matching = get_panos_of_road_and_indentify_lane_type_by_id(road_id, df_tmp, False) 
            matching = matching[atts].merge(df_tmp[['s', 'e']], left_on='osm_road_index', right_index=True)
        
        if matching.shape[0] == 0:
            print( f"{sys._getframe(0).f_code.co_name} {road_id}, no matching recods" )
            return None, None
    except:
        print( f"{sys._getframe(0).f_code.co_name} {road_id}, process error" )
        return None, None
    
    rids = []
    for i in  matching.RID.values:
        if i in rids:
            continue
        rids.append(i)
    rids_ordered = CategoricalDtype(rids, ordered=True)

    # filter outlier -> 计算路段的统计属性
    points = DB_panos.query( f"RID in {rids}" ).dropna()
    tmp = points.groupby('RID').apply( lambda x: _panos_filter(x) ).drop(columns='RID').reset_index()
    
    if outlier_filter and tmp.shape[0] != 0:
        if verbose: 
            origin_size = tmp.shape[0]
            
        _mean, _std = tmp.lane_num.mean(), tmp.lane_num.std()
        if not np.isnan(_mean) and not np.isnan(_std):
            iterverl = (_mean-mul_factor*_std, _mean+mul_factor*_std)
            tmp.query( f" {iterverl[0]} < lane_num < {iterverl[1]}", inplace=True )
            if verbose: 
                print( f"{sys._getframe(0).f_code.co_name} outlier_filter, size: {origin_size} -> {tmp.shape[0]}")
          
    if tmp.shape[0] == 0:
        print( f"{sys._getframe(0).f_code.co_name} {road_id}, no matching records after filter algorithm" )
        return None, None
    
    # reorder the panos
    tmp.loc[:, 'RID'] = tmp['RID'].astype(rids_ordered)
    tmp.sort_values(by=['RID', 'Order'], inplace=True)
    tmp.reset_index(drop=True, inplace=True)

    if offset:
        tmp.loc[:, 'lane_num'] = tmp.loc[:, 'lane_num'] - 1
        
    if vis:
        fig, ax = map_visualize(tmp, scale=.1, color='gray', figsize=(15, 15))
        df_edges.query(f'rid =={road_id}').plot(ax=ax, linestyle='--', color='black', label='OSM road', alpha=.5)
        tmp.loc[:, 'lane_num_str'] = tmp.loc[:, 'lane_num'].astype(str)
        tmp.plot(ax=ax, column='lane_num_str', legend=True)
        
        _mean, _std = tmp.lane_num.mean(), tmp.lane_num.std()
        iterverl = (round(_mean-mul_factor*_std, 1), round(_mean+mul_factor*_std,1) )
        ax.set_title(f"{road_id}, {road_name}, mean {_mean:.1f}, std {_std:.1f}, {iterverl}", fontsize=18)
        if debug:
            try:
                fig.savefig(f'../cache/matching_records/{road_name}_{road_id}.jpg', dpi=300)
            except:
                print(road_name, road_id)
        plt.tight_layout(pad=0.1)
        plt.close()
        
        return tmp, fig
        
    return tmp, None


#%%
if __name__ == '__main__':
    """ 预测指定 rid 的道路, 并输出匹配情况和照片 """
    osm_rid = 208128052 # 529249851
    mathcing_res, fig = get_and_filter_panos_by_osm_rid(osm_rid, df_edges, vis=True, debug=True)
    fig
    
    """ MatchingPanos """
    mathing_pano = MatchingPanos(df_edges)
    # single matching
    osm_rid = 208128052
    mathing_pano.matching(osm_rid)
    # batch matching
    rids = [208128052]
    mathing_pano.matching_lst(rids, vis=True, debug=True)

    """ get rids in a speaicl area by OSM_Net """
    from utils.log_helper import LogHelper, logbook
    g_log_helper = LogHelper(log_dir="/home/pcl/traffic/RoadNetworkCreator_by_View/log", log_name='sumo.log')
    SUMO_LOG = g_log_helper.make_logger(level=logbook.INFO)

    from sumo_helper import Sumo_Net, OSM_Net
    osm_net = OSM_Net(file='./osm_bbox.osm.bak.xml', save_fn='./osm_bbox.osm.xml', logger=SUMO_LOG)

    road_types_lst = ['primary', 'secondary'] # 'trunk', 
    rids = osm_net.get_rids_by_road_levels(road_types_lst)


# %%

