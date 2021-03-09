import os
import cv2 
import requests
import numpy as np
import time

def lstr_pred(fn):
    """[summary]

    Args:
        fn (str): 图片名称，不含前缀路径

    Returns:
        [type]: [description]
    """

    fn = fn.split('/')[-1] if '/' in fn else fn
    data = {
        "enable_tags": [123],
        "alg_service_uuid": "df1b658d-2659-4eab-9c30-b75173028cec",
        "is_async": False,
        "input_params": fn,
        "input_type": 1,
        "alg_service": 25,
        # "extram_params": "",
        # 'result_callback': 'http://192.168.202.92:7109/api/task-record/277/'
    }

    # process by the lstr
    r = requests.post(url='http://192.168.202.92:7109/api/task-record/', json=data)
    res = {'result_callback': r.json()['result_callback']}
    print(fn, r.status_code, r.json()['result_callback'], '...')
    
    while True:
        time.sleep(1)
        r = requests.get( res['result_callback'])
        tmp = r.json()['result']
        if tmp is not None:
            res['respond'] = tmp
            break

    return res


def draw_pred_lanes_on_img(pred_dict, out_path, root_folder = '/home/pcl/Data/minio_server/panos/'):
    """draw predited lanes on the input imgs"""

    assert 'name' in pred_dict and 'pred' in pred_dict, "dict not include 'file' or 'pred' "
    img  = cv2.imread( os.path.join( root_folder, pred_dict['name'].split('/')[-1] ))
    try:
        img_h, img_w, _ = img.shape        
    except:
        return False
    
    overlay = img.copy()
    color = (0, 255, 0)

    for i, lane in enumerate(pred_dict['pred']):
        lane = lane[1:]  # remove conf
        lower, upper = lane[0], lane[1]
        lane = lane[2:]  # remove upper, lower positions

        # generate points from the polynomial
        ys = np.linspace(lower, upper, num=50)
        points = np.zeros((len(ys), 2), dtype=np.int32)
        points[:, 1] = (ys * img_h).astype(int)
        points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys - lane[5]) * img_w).astype(int)
        points = points[(points[:, 0] > 0) & (points[:, 0] < img_w)]

        # draw lane with a polyline on the overlay
        for current_point, next_point in zip(points[:-1], points[1:]):
            overlay = cv2.line(overlay, tuple(current_point), tuple(next_point), color=color, thickness=15)

        # draw lane ID
        if len(points) > 0:
            cv2.putText(img, str(i), tuple(points[len(points)//2]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=color,
                        thickness=3)
    
    # Add lanes overlay
    w = 0.6
    img = ((1. - w) * img + w * overlay).astype(np.uint8)

    if out_path is not None:
        cv2.imwrite(out_path, img)
  
    return True


if __name__ == '__main__':
    fn = '7ea73e-734a-be1a-b9f4-2310d5_00_09005700011601081043548508N_179.jpg'
    
    res = lstr_pred(fn)

    fn = '/Data/minio_server/panos/42fe8d-9555-1595-7843-afe68c_00_09005700121709091540114199Y_271.jpg'

    res = lstr_pred(fn)


    if 'respond' in res:
        draw_pred_lanes_on_img(res['respond'], './test.jpg')
    
    