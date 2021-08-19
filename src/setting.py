""" Global config """
DIS_FACTOR = 1/110/1000
DEBUG_FOLDER = "../debug"
PANO_FOLFER = '/home/pcl/Data/minio_server/panos'

GBA_BBOX = [112.471628,  22.138605, 114.424664,  23.565487]
SZ_BBOX  = [113.746280,  22.441466, 114.623972,  22.864722]
PCL_BBOX = [113.931914,  22.573536, 113.944456,  22.580613]
LXD_BBOX = [113.92423,   22.57047,  113.94383,   22.58507]
FT_BBOX  = (114.05097,   22.53447,  114.05863,   22.54605)
SZU_BBOX = (113.92370,   22.52889,  113.94128,   22.54281)


""" road_type_filter """
# Note: we adopt the filter logic from osmnx (https://github.com/gboeing/osmnx)
# exclude links with tag attributes in the filters
filters = {}

filters['auto'] = {'area':['yes'],
                   'highway':['cycleway','footway','path','pedestrian','steps','track','corridor','elevator','escalator',
                              'proposed','construction','bridleway','abandoned','platform','raceway','service'],
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
