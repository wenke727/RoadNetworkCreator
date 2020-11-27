import ctypes
from ctypes import cdll
import os


# module coordination transfer
lib = cdll.LoadLibrary( os.path.join(os.path.dirname(__file__), 'baiduCoord.so'))

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

if __name__ == '__main__':
    x, y = bd_coord_to_mc(113.949221,22.545245)
    pass    