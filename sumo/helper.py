from pyproj import CRS, Transformer


def proj_trans(x, y, in_sys=4326, out_sys=32649, offset=(-799385.77,-2493897.75), precision=2 ):
    """proj trans

    Args:
        x ([type]): [description]
        y ([type]): [description]
        in_sys (int, optional): [description]. Defaults to 4326.
        out_sys (int, optional): [description]. Defaults to 32649.
        offset (tuple, optional): [description]. Defaults to (-799385.77,-2493897.75).
        precision (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    # assert not isinstance(coord, tuple) and len(coord) != 2, "check coord"
    
    # always_xy (bool, optional) – If true, the transform method will accept as input and return as output coordinates using the traditional GIS order, that is longitude, latitude for geographic CRS and easting, northing for most projected CRS. 
    coord_transfer = Transformer.from_crs( CRS(f"EPSG:{in_sys}"), CRS(f"EPSG:{out_sys}"), always_xy=True )
    x, y = coord_transfer.transform(x, y)
    x += offset[0]
    y += offset[1]
    
    return round(x, precision), round(y, precision)


def df_coord_transfor(gdf):
    # 坐标转换算法
    # <location netOffset="-799385.77,-2493897.75" convBoundary="0.00,0.00,18009.61,5593.04" origBoundary="113.832744,22.506539,114.086290,22.692155" projParameter="+proj=utm +zone=49 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"/>

    crs = CRS("+proj=utm +zone=49 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    to_crs = crs.to_epsg()

    gdf.to_crs(epsg=to_crs, inplace = True)

    gdf.loc[:, "x_"] = gdf.geometry.x - 799385.77
    gdf.loc[:, "y_"] = gdf.geometry.y - 2493897.75

    return gdf
