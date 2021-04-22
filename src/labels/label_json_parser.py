#%%
import os
import json
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
from label_helper import resize_pano_img_for_training


Y_MIN  = 240
Y_MAX  = 720 - 10
Y_AXIS = pd.DataFrame( np.linspace( 240, Y_MAX, (Y_MAX - Y_MIN)//10+1 ).astype(np.int), columns=['y'] )

pano_dir = "/home/pcl/Data/minio_server/panos"
save_path = '/home/pcl/Data/LaneDetection'

def make_dirs(save_path):
    for i in ['gt', 'clips']:
        path = os.path.join( save_path, i)
        if not os.path.exists(path):
            os.makedirs(path)


def normalize_naming(label_dir = '/home/pcl/traffic/data/1st_batch',
                  pano_dir = pano_dir,
                  label_remove_dir = None
                  ):
    labels_remove = set([ x.replace('.jpg', '') for x in  os.listdir(label_remove_dir)]) if label_remove_dir is not None else set()

    label_lst = [] 
    for f in os.listdir(label_dir):
        if f.replace('.json', '') in labels_remove:
            continue
        fn = "_".join(f.split("_")[-4:])
        label_lst.append(fn)
        if fn != f:
            os.rename( os.path.join(label_dir, f), os.path.join(label_dir, fn) )

    print( "label_lst: ", len(label_lst))
    
    return label_lst


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
        # f1 = np.polyfit( y, x, 3 )
        # p1 = np.poly1d(f1)

        upper, lower = int(y.max()/10)*10, int(y.min()/10)*10
        pred_y = np.linspace(lower, upper, num = int((upper-lower)/10 + 1) )
        # pred_x = p1(pred_y)
        if y[0] > y[-1]:
            pred_x = np.interp( pred_y, y[::-1], x[::-1] )
        else:
            pred_x = np.interp( pred_y, y, x )
            
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

        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.margins(0,0)
        if fn is not None:
            plt.savefig( fn, pad_inches=0, bbox_inches='tight' , dpi=160 )
        plt.close()
        return 


def scp_to_remote_server_89():
    password = 'root'
    user     = 'root'
    ip       = '192.168.203.89'
    dst      = '/root/TuSimple'
    
    res = os.popen( f" sshpass -p {password} scp -P 7022 -r {save_path} {user}@{ip}:{dst}" ).read()
    
    return     


def label_process_batch( origin_img=True, write_label_to_img=True):
    # rename label
    for fn in os.listdir( label_dir ):
        lst = fn.split("_")
        if len(lst) <= 4:
            continue
        os.rename( os.path.join( label_dir, fn ), os.path.join( label_dir, "_".join(lst[-4:]) ) )

    # resize pano img to the training size
    f_lst = os.listdir(label_dir)
    if origin_img:
        for label_file in tqdm(f_lst, 'move and resize imgs: '):
            fn = label_file.split(".")[0] + ".jpg"
            # print(os.path.join( save_path, 'clips', fn ))
            resize_pano_img_for_training( os.path.join( pano_dir, fn ), os.path.join( save_path, 'clips', fn ))

    # transfer lables
    res = []
    error_lst = []
    for label_file in tqdm(f_lst[:10], "tranfer labels"):
        fn = label_file.split(".")[0]
        try:
            label_pano = Lane_label( fn, os.path.join( save_path, 'clips'), label_dir )
            record = label_pano.label_to_json()
            res.append(record)
            
            if write_label_to_img: label_pano.plot( os.path.join( save_path, 'gt', fn) )
            del label_pano

        except:
            error_lst.append(fn)

    with open(f'{save_path}/label_data_nanshan_0126.json', 'w') as f:
        f.write( '\n'.join([ i.replace(f"{save_path}/", '') for i in  res[:-500]])+"\n" )

    with open(f'{save_path}/label_data_nanshan_0126_val.json', 'w') as f:
        f.write( '\n'.join([ i.replace(f"{save_path}/", '') for i in  res[-500:]])+"\n" )


def label_process( f_lst, origin_img=True, write_label_to_img=True):
    # rename label
    for fn in f_lst:
        lst = fn.split("_")
        if len(lst) <= 4:
            continue
        os.rename( os.path.join( label_dir, fn ), os.path.join( label_dir, "_".join(lst[-4:]) ) )

    # resize pano img to the training size
    if origin_img:
        for label_file in f_lst:
            fn = label_file.split(".")[0] + ".jpg"
            print(os.path.join( pano_dir, fn ))
            resize_pano_img_for_training( os.path.join( pano_dir, fn ), os.path.join( save_path, 'clips', fn ))

    # transfer lables
    res = []
    error_lst = []
    for label_file in f_lst:
        fn = label_file.split(".")[0]
        try:
            label_pano = Lane_label( fn, os.path.join( save_path, 'clips'), label_dir )
            record = label_pano.label_to_json()
            res.append(record)
            
            if write_label_to_img: label_pano.plot( os.path.join( save_path, 'gt', fn) )
            del label_pano

        except:
            error_lst.append(fn)

    return res


def label_process_parrallel(label_dir, name):
    global save_path
    num = 50
    label_lst = os.listdir( label_dir ) 
    data = pd.DataFrame(label_lst, columns=['pid'])
    data_grouped = data.groupby(data.index/num)
    results = Parallel(n_jobs=num)(delayed(label_process)(group.pid.values) for name, group in data_grouped)

    res = []
    for i in results: 
        res += i

    with open(f'{save_path}/{name}.json', 'w') as f:
        f.write( '\n'.join([ i.replace(f"{save_path}/", '') for i in  res])+"\n" )

    # with open(f'{save_path}/label_data_nanshan_0129_val.json', 'w') as f:
    #     f.write( '\n'.join([ i.replace(f"{save_path}/", '') for i in  res[-500:]])+"\n" )


def copy_to_LSTR_docker():
    dst = "/home/pcl/Data/TuSimple"
    os.popen( f" cp -r {save_path} {dst}" )
    os.popen( f" cp {save_path} {dst}" )
    
    return


if  __name__ == '__main__':
    # label_process(os.listdir( label_dir ))
    # folder = '/home/pcl/traffic/data/1st_batch'
    # name = folder.split("/")[-1]
    # make_dirs(save_path)
    # normalize_naming(label_dir = folder)
    
    # label_process_parrallel(label_dir, name)


    # folder = '/home/pcl/traffic/data/2nd_batch'
    # name = folder.split("/")[-1]
    # make_dirs(save_path)
    # normalize_naming(label_dir = folder)
    
    # label_process_parrallel(folder, name)



    folder = label_dir = '/home/pcl/traffic/data/2nd_batch_edge_coverage'
    name = folder.split("/")[-1]
    make_dirs(save_path)
    normalize_naming(label_dir = folder)
    
    label_process_parrallel(folder, name)



    folder = label_dir = '/home/pcl/traffic/data/2nd_batch_zebra_crossing'
    name = folder.split("/")[-1]
    make_dirs(save_path)
    normalize_naming(label_dir = folder)
    
    label_process_parrallel(folder, name)

    # copy_to_LSTR_docker()
    # scp_to_remote_server_89()
    
    # label_pano = Lane_label( '09005700011601091410301692P', './', './' )
    # record = label_pano.label_to_json()

    # with open('./label.json', 'w') as f:
    #     f.write(record)
    
    # label_pano.plot('./test.jpg')

    # transfer to LSTR docker

