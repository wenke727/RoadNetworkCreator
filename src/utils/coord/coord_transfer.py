import math
import coordTransform_py.CoordTransform_utils as ct
import ctypes 
import os
import math
import types

lib = ctypes.cdll.LoadLibrary( os.path.join(os.path.dirname(__file__), 'baiduCoord.so'))

class coords_type(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double), ('y', ctypes.c_double)]

LL2MC_lng = lib.LL2MC_lng 
LL2MC_lat = lib.LL2MC_lat  
MC2LL_lat = lib.MC2LL_lat
MC2LL_lng = lib.MC2LL_lng
for i in [LL2MC_lng, LL2MC_lat, MC2LL_lat, MC2LL_lng]:
    i.argtypes = [ctypes.c_double, ctypes.c_double]
    i.restype  = ctypes.c_double


LL2MC, MC2LL = lib.LL2MC, lib.MC2LL 
for i in [LL2MC, MC2LL]:
    i.argtypes = [ctypes.c_double, ctypes.c_double]
    i.restype = coords_type

# LL2MC.argtypes = [ctypes.c_double, ctypes.c_double]
# LL2MC.restype  = coords_type

# MC2LL.argtypes = [ctypes.c_double, ctypes.c_double]
# MC2LL.restype  = coords_type

def bd_coord_to_mc( lng, lat ):
    coord = LL2MC(lng, lat)
    return coord.x, coord.y

def bd_coord_to_mc_old( lng, lat ):
    return LL2MC_lng(lng, lat), LL2MC_lat(lng, lat)

def bd_mc_to_coord( lng, lat ):
    coord = MC2LL(lng, lat)
    return coord.x, coord.y

def bd_mc_to_coord_old( lng, lat ):
    return MC2LL_lng(lng, lat), MC2LL_lat(lng, lat)

def bd_mc_to_wgs_vector( record, attr = ["X", "Y"], factor = 100 ):
    return ct.bd09_to_wgs84(*bd_mc_to_coord( record[attr[0]]/factor, record[attr[1]]/factor ))

def bd_mc_to_wgs(x, y, factor = 100 ):
    return ct.bd09_to_wgs84(*bd_mc_to_coord( x/factor, y/factor ))

def wgs_to_bd_mc(lng, lat):
    return  bd_coord_to_mc(*ct.wgs84_to_bd09(lng, lat))

def bd_tile_to_bd_mc(x, y, zoom):
    # Ref 百度地图根据经纬度计算瓦片行列号 https://www.cnblogs.com/xiaozhi_5638/p/4748186.html
    lng, lat = x * 256 * math.pow(2, 18-zoom), y *256* math.pow(2, 18-zoom)
    return lng, lat

if __name__ == "__main__":
    
    print(bd_tile_to_bd_mc(49543, 10090, 18))

    bd_coord_to_mc(113, 22)
    bd_coord_to_mc_old(113, 22)

    pass
