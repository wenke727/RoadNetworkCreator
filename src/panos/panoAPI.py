import os
import requests
import time

def get_panos(params):
    """Crawled panos by `pano_distributed_crawler`.
    location: /home/pcl/traffic/pano_distributed_crawler
    github: https://git.pcl.ac.cn/huangwk/pano_distributed_crawler

    Args:
        fn (str): 图片名称，不含前缀路径

    Returns:
        [type]: [description]
    """
    url = 'http://192.168.202.92:7109/api/start-task/'
    data = {
        "enable_tags": [123],
        "alg_service_uuid": "95c3b5a1-a6ec-4f53-aa6f-70034871b9fe",
        "is_async": False,
        "input_params": str(params),
        "input_type": 1,
        # "alg_service": 29,
        # "extram_params": "",
    }

    # process by the lstr
    r = requests.post(url=url, json=data)
    res = {'result_callback': r.json()['result_callback']}
    # print( r.status_code, r.json()['result_callback'], '...')
    
    return res

if __name__ == '__main__':
    params = {'pid': '01005700001312021447154435T',
            'heading': 202,
            'path': '../download/9e7a5c-a72d-1625-bed6-b74eb6_15_01005700001312021447154435T_202.jpg'
        }
    res = get_panos( params )