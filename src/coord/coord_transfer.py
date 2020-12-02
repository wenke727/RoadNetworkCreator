import coordTransform_py.CoordTransform_utils as ct
import ctypes 
import os

lib = ctypes.cdll.LoadLibrary( os.path.join(os.path.dirname(__file__), 'baiduCoord.so'))

LL2MC_lng = lib.LL2MC_lng 
LL2MC_lat = lib.LL2MC_lat  
LL2MC_lng.argtypes = [ctypes.c_double, ctypes.c_double]
LL2MC_lat.argtypes = [ctypes.c_double, ctypes.c_double]
LL2MC_lng.restype  = ctypes.c_double
LL2MC_lat.restype  = ctypes.c_double

MC2LL_lat = lib.MC2LL_lat
MC2LL_lng = lib.MC2LL_lng
MC2LL_lat.argtypes = [ctypes.c_double, ctypes.c_double]
MC2LL_lng.argtypes = [ctypes.c_double, ctypes.c_double]
MC2LL_lat.restype  = ctypes.c_double
MC2LL_lng.restype  = ctypes.c_double

def bd_coord_to_mc( lng, lat ):
    return LL2MC_lng(lng, lat), LL2MC_lat(lng, lat)

def bd_mc_to_coord( lng, lat ):
    return MC2LL_lng(lng, lat), MC2LL_lat(lng, lat)

def bd_mc_to_wgs_vector( record, attr = ["X", "Y"], factor = 100 ):
    return ct.bd09_to_wgs84(*bd_mc_to_coord( record[attr[0]]/factor, record[attr[1]]/factor ))

def bd_mc_to_wgs(x, y, factor = 100 ):
    return ct.bd09_to_wgs84(*bd_mc_to_coord( x/factor, y/factor ))



