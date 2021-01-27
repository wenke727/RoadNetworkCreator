from shapely import geometry
from geopandas import gpd
import urllib
import matplotlib.pyplot as plt
# from main import *
from PIL import Image

route = gpd.read_file( "../output/for_Presentation_留仙洞.geojson" )
points = gpd.GeoDataFrame(route[['RID']].merge(DB_panos, on='RID'))

# points.query( "DIR != 0" ).reset_index().to_file( '../output/points_liuxiandong_presetation.geojson', driver="GeoJSON" )
points.query( "DIR != 0", inplace=True )

points.info()



def draw_polygon_by_bbox(bbox=[113.93306,22.57437, 113.9383, 22.58037]):
    # extract the roads of intrest
    from shapely.geometry import LineString

    coords = [bbox[:2], [bbox[0], bbox[3]],
              bbox[2:], [bbox[2], bbox[1]], bbox[:2]]

    area = gpd.GeoDataFrame( [{'name': 'presetation area', 'geometry':LineString(coords)}] )
    return area


def get_staticimage(id, heading, folder=pano_dir):
    file_name = f"{folder}/{id}.jpg"
    if os.path.exists(file_name):
        return Image.open(file_name)

    # print(file_name)
    # id = "09005700121902131650290579U"; heading = 87
    url = f"https://mapsv0.bdimg.com/?qt=pr3d&fovy=88&quality=100&panoid={id}&heading={heading}&pitch=10&width=1024&height=1024"
    request = urllib.request.Request(url=url, method='GET')
    map = urllib.request.urlopen(request)

    f = open(file_name, 'wb')
    f.write(map.read())
    f.flush()
    f.close()
    return Image.open(file_name)



def plot_pano_position(pid):
    lxd_area = draw_polygon_by_bbox()
    # attemp 1
    fig, ax = map_visualize( lxd_area, lyrs='y', scale=-0.05, color='red',figsize=(4,6))
    # 去除坐标轴
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    # ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_major_locator(plt.NullLocator())
    # 去除黑框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    points.query( f"PID=='{pid}' " ).plot(ax=ax, color='red')
    # 移除白边
    fig.savefig('./position.jpg', dpi=300, bbox_inches = 'tight', pad_inches=0.0)
    return Image.open('./position.jpg')

# point_presentation = plot_pano_position(pid = '09005700121709091540516459Y')

def plot_postion_and_pano(pid, direction, save_path=None):
    # 双栏显示
    point_presentation = plot_pano_position(pid)
    img = get_staticimage(pid, direction)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(hspace=0, wspace=0)

    # axes[0].imshow( Image.open('./position.jpg').resize((512,512)) )
    # axes[1].imshow( img.resize((512, 512)) )
    axes[0].imshow( point_presentation )
    axes[1].imshow( img )
    for ax in axes:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout(pad =0, h_pad=0, w_pad=0.1)
    if save_path is not None:
        fig.savefig( save_path, dpi=500,  bbox_inches = 'tight', pad_inches=0.0 )



for index, (pid, dir) in enumerate( points[['PID','DIR']].values[40:]):
    print(pid, dir)
    plot_postion_and_pano( pid, dir, f"../output/presentation/{index}.jpg")



