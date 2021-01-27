import os
import json
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from label_helper import resize_pano_img_for_training


Y_MIN  = 240
Y_MAX  = 720 - 10
Y_AXIS = pd.DataFrame( np.linspace( 240, Y_MAX, (Y_MAX - Y_MIN)//10+1 ).astype(np.int), columns=['y'] )

pano_dir = "/home/pcl/Data/minio_server/panos"
label_dir = './label_data'
save_path = "/home/pcl/Data/TuSimple/LaneDetection"

class Lane_label():
    def __init__(self, pid, img_folder, json_folder, pic_format='jpg', factor=1280/1024):
        self.pid = pid
        self.img_folder = img_folder
        self.img_path  = os.path.join( img_folder, f"{pid}.{pic_format}" )
        self.json_path = os.path.join( json_folder, f"{pid}.json" )
    
        with open(self.json_path) as f:
            labels = json.loads( f.read() )
        self.df = pd.DataFrame(labels['shapes'])
        self.df.loc[:, 'points'] = self.df.apply( lambda x: (np.array(x.points)*factor).astype(np.int), axis=1 )
        self.df.loc[:, 'lines'] = self.df.points.apply( lambda x: self.poly_fitting(x) )
        pass

    def poly_fitting(self, line):
        x, y = line[:,0], line[:,1]
        f1 = np.polyfit( y, x, 3 )
        p1 = np.poly1d(f1)

        upper, lower = int(y.max()/10)*10, int(y.min()/10)*10
        pred_y = np.linspace(lower, upper, num = int((upper-lower)/10 + 1) )
        pred_x = p1(pred_y)

        df = Y_AXIS.merge(pd.DataFrame( {'x': pred_x, 'y': pred_y} ), on='y', how='outer').fillna(-2).astype(np.int)
        # Warming: 需要统一尺寸, height大于720的直接删除
        df.query( f" y <= {Y_MAX} ", inplace=True )
        
        return df[['x','y']].values

    def label_to_json(self, ):
        """return the labels in the format of Tusimple"""
        label_info = {
            "lanes"     : self.df.lines.apply( lambda x: list(x[:, 0]) ).values.tolist(),
            "h_samples" : Y_AXIS['y'].values.tolist(),
            "raw_file"  : os.path.join('clips', f"{self.pid}.jpg")
            # "raw_file"  : os.path.join(self.img_folder, f"{self.pid}.jpg")
        }

        return str( label_info ).replace("'", "\"")

    def plot(self, fn=None ):
        img = Image.open(self.img_path)
        
        plt.imshow(img)
        for l in self.df.lines:
            l = np.array( [ [x, y] for x, y in l if x >= 0 ])
            plt.plot( l[:,0], l[:,1], '.' )

        # plt.ylim( img.size[1] )
        plt.axis('off')
        # plt.xlim( 0, img.size[0] )
        plt.tight_layout(pad=0)
        plt.margins(0,0)
        if fn is not None:
            plt.savefig( fn, pad_inches=0, bbox_inches='tight' , dpi=200 )

def main():
    
    # rename label
    for fn in os.listdir( label_dir ):
        lst = fn.split("_")
        if len(lst) <= 4:
            continue
        os.rename( os.path.join( label_dir, fn ), os.path.join( label_dir, "_".join(lst[-4:]) ) )

    # resize pano img to the training size
    f_lst = os.listdir(label_dir)
    for label_file in tqdm(f_lst, 'move and resize imgs: '):
        fn = label_file.split(".")[0] + ".jpg"
        # print(os.path.join( save_path, 'clips', fn ))
        resize_pano_img_for_training( os.path.join( pano_dir, fn ), os.path.join( save_path, 'clips', fn ))

    # transfer lables
    res = []
    error_lst = []
    for label_file in tqdm(f_lst, "tranfer labels"):
        fn = label_file.split(".")[0]
        try:
            label_pano = Lane_label( fn,  os.path.join( save_path, 'clips'), label_dir )
            record = label_pano.label_to_json()
            res.append(record)
        except:
            error_lst.append(fn)

    with open(f'{save_path}/label_data_nanshan_0126.json', 'w') as f:
        f.write( '\n'.join([ i.replace(f"{save_path}/", '') for i in  res[:-500]])+"\n" )

    with open(f'{save_path}/label_data_nanshan_0126_val.json', 'w') as f:
        f.write( '\n'.join([ i.replace(f"{save_path}/", '') for i in  res[-500:]])+"\n" )

if  __name__ == '__main__':
    main()
        
    # label_pano = Lane_label( '09005700011601091410301692P', './', './' )
    # record = label_pano.label_to_json()

    # with open('./label.json', 'w') as f:
    #     f.write(record)
    
    # label_pano.plot('./test.jpg')





