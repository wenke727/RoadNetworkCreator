""" Global config """
DIS_FACTOR = 1/110/1000
DEBUG_FOLDER = "../debug"
CACHE_FOLDER = "../cache"
PANO_FOLFER = '/home/pcl/Data/minio_server/panos'

PRED_MEMO = "/home/pcl/Data/minio_server/input/lane_shape_predict_memo.h5"


GBA_BBOX = [112.471628,  22.138605, 114.424664,  23.565487]
SZ_BBOX  = [113.746280,  22.441466, 114.623972,  22.864722]
PCL_BBOX = [113.931914,  22.573536, 113.944456,  22.580613]
LXD_BBOX = [113.92423,   22.57047,  113.94383,   22.58507]
FT_BBOX  = [114.02874162861015, 22.52426853077481, 114.06680715668308, 22.56334823810368]
FT_samll_BBOX  = [114.05097,   22.53447,  114.05863,   22.54605]
SZU_BBOX = [113.92370,   22.52889,  113.94128,   22.54281]


""" road_type_filter """
# Note: we adopt the filter logic from osmnx (https://github.com/gboeing/osmnx)
# exclude links with tag attributes in the filters
filters = {}

filters['auto'] = {'area':['yes'],
                   'highway':['cycleway','footway','path','pedestrian','steps','track','corridor','elevator','escalator',
                              'proposed','construction','bridleway','abandoned','platform','raceway'],
                   'motor_vehicle':['no'],
                   'motorcar':['no'],
                   'access':['private'],
                   'service':['parking','parking_aisle','driveway','private','emergency_access']
                   }

filters['bike'] = {'area':['yes'],
                   'highway':['footway','steps','corridor','elevator','escalator','motor','proposed','construction','abandoned','platform','raceway'],
                   'bicycle':['no'],
                   'service':['private'],
                   'access':['private']
                   }

filters['walk'] = {'area':['yes'],
                   'highway':['cycleway','motor','proposed','construction','abandoned','platform','raceway'],
                   'foot':['no'],
                   'service':['private'],
                   'access':['private']
                   }


"""" road_level """
link_type_no_dict = {
    'motorway':1, 
    'trunk':2, 
    'primary':3, 
    'secondary':4, 
    'tertiary':5, 
    'motorway_link':5.1, 
    'trunk_link':5.2, 
    'primary_link':5.3, 
    'secondary_link':5.4, 
    'tertiary_link':5.5, 
    'residential':6, 
    'service':7, 
    
    'cycleway':8, 
    'footway':9, 
    'track':10, 
    'unclassified':11, 
    'living_street': 15, 
    'connector':20, 
    'railway':30, 
    'aeroway':31
}
