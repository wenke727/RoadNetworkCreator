# Ref: /home/pcl/traffic/map_factory/ImageRelatedProcess.py
from PIL import Image
import os


def merge_tiles(f_lst, file_name_catalog = True, to_fn=None):
    """
    merge tiles into one
    @param: f_lst: tiles list
    @param: the catalog type that store in the system
    @return: the merge tile
    """
    cha = "_"
    xs, ys = [], []
    z =  f_lst[0].split(cha)[:-2]
    
    z_folder =  '/'.join( f_lst[0].split('/')[:-2])

    if file_name_catalog:
        for filename in f_lst:
            items = filename.replace('.jpg', '').split(cha)
            z, x, y = [int(i) for i in items[-3:]]
            if x not in xs:
                xs.append(x)
            if y not in ys:
                ys.append(y)

    # 定义图像拼接函数
    IMAGE_SIZE = 256
    max_x, min_x, max_y, min_y = max(xs), min(xs), max(ys), min(ys)
    IMAGE_COLUMN = max_x - min_x + 1
    IMAGE_ROW    = max_y - min_y + 1
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))

    for x in xs:
        for y in ys:
            path = os.path.join( z_folder, f'./imgs/{z}_{x}_{y}.jpg')
            try:
                from_image = Image.open( path ).resize( (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                to_image.paste( from_image, ((x - min_x ) * IMAGE_SIZE, (y - min_y) * IMAGE_SIZE) )
            except:
                print(f"数据缺失: {path}")
                pass

    to_image.save( "merge.jpg" if to_fn is None else to_fn)
    
    return to_image


merge_tiles( os.listdir("./imgs"), to_fn = "log01_dink34-1.4-2.5.jpg" )


