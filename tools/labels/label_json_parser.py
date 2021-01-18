import json
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

Y_MIN  = 240
Y_MAX  = 720 - 10
Y_AXIS = pd.DataFrame( np.linspace( 240, Y_MAX, (Y_MAX - Y_MIN)//10+1 ).astype(np.int), columns=['y'] )


class Lane_label():
    def __init__(self, pid, img_folder, json_folder, pic_format='jpg'):
        self.pid = pid
        self.img_folder = img_folder
        self.img_path  = os.path.join( img_folder, f"{pid}.{pic_format}" )
        self.json_path = os.path.join( json_folder, f"{pid}.json" )
    
        with open(self.json_path) as f:
            labels = json.loads( f.read() )
        self.df = pd.DataFrame(labels['shapes'])
        self.df.loc[:, 'points'] = self.df.apply( lambda x: np.array(x.points).astype(np.int), axis=1 )
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
            'h_samples' : Y_AXIS['y'].values.tolist(),
            'lanes'     : self.df.lines.apply( lambda x: list(x[:, 0]) ).values.tolist(),
            'raw_file'  : self.pid
        }

        return str( label_info )

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


if  __name__ == '__main__':
        
    label_pano = Lane_label( '09005700011601091410301692P', './', './' )
    label_pano.label_to_json()
    label_pano.plot('./test.jpg')
