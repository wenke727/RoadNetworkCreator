import os, io, sys
import copy
import cv2
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from PIL import Image
import math

# from db.db_process import load_from_DB
sys.path.append(os.path.dirname(__file__))
from geo_plot_helper import map_visualize
from df_helper import query_df
# from utils.spatialAnalysis import linestring_length
# from utils.utils import load_config


""" import road network from OSM """
# import pickle

# config = load_config()
# pano_dir = config['data']['pano_dir']
# DF_matching = pd.read_csv( config['data']['df_matching'])


def plot_pano_and_its_view(pid, DB_panos, DB_roads, heading=None):
    """绘制pano所在的路段，位置以及视角

    Args:
        pid ([type]): [description]
    """
    rid = DB_panos.query( f"PID=='{pid}' " ).RID.iloc[0]
    pid_record = query_df(DB_panos, "RID", rid).query( f"PID == '{pid}'" )
    assert( len(pid_record) > 0 )
    pid_record = pid_record.iloc[0]

    if heading is None:
        heading = pid_record.DIR
    x, y = pid_record.geometry.coords[0]
    
    fig, ax = map_visualize( DB_roads.query( f"RID == '{pid_record.RID}' " ), label="Lane" )

    x0, x1 = ax.get_xlim()
    aus_line_len = (x1-x0)/20
    dy, dx = math.cos(heading/180*math.pi) * aus_line_len, math.sin(heading/180*math.pi) * aus_line_len
    ax.annotate('', xy=(x+dx, y+dy), xytext= (x,y) ,arrowprops=dict(facecolor='blue', shrink=0.05, alpha=0.5))
    gpd.GeoSeries( [Point(x, y)] ).plot(ax=ax, label='Pano', marker='*',  markersize= 360 )

    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    return fig


#! CV2 celated
def url_to_image_CV2(url='http://www.pyimagesearch.com/wp-content/uploads/2015/01/opencv_logo.png', type=''):
    import numpy as np
    import cv2
    from urllib.request import urlopen

    # download the image, convert it to a NumPy array, and then read it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    return image


def Image_to_CV2(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  


def images_to_video(path):
    input_folder = "./detections"

    fns = os.listdir(input_folder)

    img = cv2.imread( os.path.join(input_folder, fns[0]) )
    x, y, z = img.shape
    img_array = []
 
    for f in fns:
        img = cv2.imread(os.path.join(input_folder, f))
        if img is None:
            print(f + " is error!")
            continue
        img_array.append(img)
    
    len(img_array)
    # 图片的大小需要一致
    # img_array, size = resize(img_array, 'largest')
    
    fps = 1
    out = cv2.VideoWriter('demo.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (x,y ), True)
 
    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
    cv2.destroyAllWindows()


# ! Image relared
def plt_2_Image(fig):
    # method 1: PIL -> Image
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg',pad_inches=0, bbox_inches='tight', )
    buf.seek(0)
    img_new = copy.deepcopy(Image.open(buf))
    buf.close()

    # method 2: PIL -> Image
    # img_new = fig2data(fig)
    return img_new  


def combine_imgs(imgs):
    """merge imgs by row.

    Args:
        imgs ([type]): [description]

    Returns:
        [type]: [description]
    """
    n = len(imgs)
    if n == 0: return None
    
    if not isinstance(imgs[0], Image.Image):
        imgs = [ Image.open(i) for i in imgs ]
    
    heights = [ img.size[1] for img in imgs ]
    
    w, _ = imgs[0].size
    to_img = Image.new('RGB', ( w, sum(heights)))

    acc_h = 0
    for index, i in enumerate(imgs):
        to_img.paste( i, (0, acc_h ) )
        acc_h += heights[index]

    return to_img


def cv2_2_Image(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  


def fig_2_Image(fig):
    """
    Ref: https://panjinquan.blog.csdn.net/article/details/104179723
    
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # fig.subplots_adjust(top=0, bottom=0, left=0, right=0, hspace=0, wspace=0)
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombuffer("RGBA", (w, h), buf.tobytes())
    return image



if __name__ == '__main__':
    
# 在街景中添加位置示意图

    pid = '09005700121709091541462499Y'
    position = plot_pano_and_its_view( pid = '09005700121709091541462499Y' )
    position.savefig('./test.jpg', pad_inches=0, bbox_inches='tight')
    position = Image.open('./test.jpg')


    img = Image.open('/home/pcl/Data/minio_server/panos/989d83-aa81-9df2-b360-685876_02_09005700121709091541462499Y_269.jpg')
    # 将一张图粘贴到另一张图像上
    x, y = [ int(x/3) for x in img.size]
    x = int(position.size[1] *y/position.size[0])

    location_illustration = position.resize((x, y))
    img.paste( location_illustration, [0,0,x,y] )
    img


    plt.imshow(img)










