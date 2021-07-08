# RoadNetworkCreator_by_View

## 操作手册

- 街景数据获取及预测更新

  ``` python
  # src/panos_traverse.py
  crawel_panos_in_district_area('盐田区')
  
  # Docker(lstr): /Data/LaneDetection_PCL/LSTR/predict.py
  predict_batch('/Data/minio_server/panos', overwrite=False)
  
  # db/db_process.py: update lane num in panos and roads
  update_lane_num_in_DB()
  ```

## 路网生成

通过API获取深圳市osm路网

### 脚本简介

- src/baidu_map.py
  和百度地图相关的脚本，用于获取道路线性的信息

  - get_road_shp_by_search_API
    get road shape by Baidu searching API using the road name
  - roads_from_baidu_search_API
    借助fishnet，从`百度地图`中获取路网
  - fishenet
    create fishnet based on the polygon

- src/pano_base.py
  本脚本主要用于`采集pano基础信息`，实现的方法是通过bfs遍历道路，
  其中也涉及`get_unvisited_point`获取没有覆盖的点

  - get_proxy
    获取代理资源
  - get_road_buffer
    Obtain the buffer of a special road by name. The shape is queried by the Baidu searching API.
  - query_pano
    通过（x,y）或者pid获取pano的信息
  - traverse_panos_by_road_name
    Traverse Baidu panos through the road name. This Function would query the geometry by searching API. Then matching algh is prepared to matched the panos
    to the geometry.

- src/pano_img.py
  通过DB中的`线数据`获取`pano照片`

  - get_staticimage
    get static image from Baidu View with `pano id`. And there is a distributed version: <https://git.pcl.ac.cn/huangwk/pano_distributed_crawler>
  - traverse_panos_by_rid
    通过rid获取该路段上所有的pano数据

- src/panos_traverse.py
  本脚本

  - traverse_panos_by_rid
    obtain the panos in road[rid], 若没有数据则调用pano_API 异步获取数据
  - count_panos_num_by_area
    Count panos number in several area.
  - crawel_panos_in_district_area
    用于查询图片是否已抓取，若没有则调用`panoAPI`开启异步模式
    获取某一行政区内所有的街景图片

- src/predict_lanes.py
  主要用于某个路段`车道线预测可视化`的情况，以及是否添加方位角、地图、线形等元素。

  - pred_osm_road_by_rid
    基于osm中rid的道路`匹配panos`，然后调用模型预测车道线情况，将预测结果绘制在街景上。其中调用子函数`lane_shape_predict_for_rid_segment`(预测某一个路段所有的街景，并生成预测动画或图片), 然后将所有rid的照片合并成一张图片
  - plot_pano_and_its_view, add_location_view_to_img
    Plot helper
  - update_unpredict_panos
    预测尚未有数据的街景照片

- src/road_matching.py
  主要
  - get_panos_of_road_by_id
    通过frenchet距离匹配某条道路的百度街景pano轨迹，并返回匹配的pano
  - .
    ..

- src/road_network.py
  主要
  - .
    ..
  - .
    ..

<!-- 
* src/.py
  本脚本

  * .
    ..
  * .
    ..
  * .
    ..
  * .
    .. 
-->

----

## API

### 视网膜系统

启动

``` bash
# physical machine
cd /home/pcl/server/pcl/pano_data
nohup python3 pano.py > pano.log &
```

API情况

- pano_data
通过bbox获取矩阵范围内的点元素和线元素
<http://192.168.135.15:4000/get_pano_data?feature=line&bbox=113.929807,22.573702,113.937680,22.578734>

- 获取临近点
给定一个点坐标（lon和lat），返回临近的点信息
<http://192.168.135.15:4000/get_nearby_panos?lon=113.93412&lat=22.575369>

- lstr_预测_by_pid
通过pid来预测车道线的情况，返回格式可设置为json或者img；
预测上传照片的功能尚未开放
<http://192.168.135.15:4000/pred_by_pid?format=img&pid=09005700122003201356273565O>

- lstr_预测_by_full_name
<http://192.168.135.15:5000/lstr?fn=91868b-159f-3c50-008c-55ec99_04_01005700001311231236314705I_253.jpg>

  ``` bash
  # 启动
  # docker lstr
  cd /Data/LaneDetection_PCL/LSTR
  nohup sh start_web_api.sh > ../web_API.log  &
  ```

批量更新处理, 预测并更新街景车道线数量

- Use the latest model to predict the lane number of each pano and update the attribute. The result is stored in `/Data/minio_server/input/lane_shape_predict_memo.csv`

  ```
  # docker lstr 
  cd /Data/LaneDetection_PCL/LSTR
  conda activate lstr
  python predict.py
  ```

----

## 数据

### 常用脚本

|File|Memo|
|--|--|
| src/db/db_process.py| `update_lane_num_in_DB`: 用于更新DB中的车道线情况 |
| src/db/test_dataset_other_cities_panos.py | 用于获取其他城市的街景用于构建测试集 |
| src/db/features_API.py | 用于按区域提取pano`点数据`和`线数据` |
| src/db/minio_helper.py | 上传文件至minio服务器 |
| src/labels/label_json_parser | 将labelme的数据转换成lstr模型的输入 |
| src/model/lstr.py | 访问`lstr api`获取街景的车道线信息 |
| src/panos/panoAPI.py | 基于celery构建的分布式爬虫接口 |
|  |  |

### 分布式街景爬虫

借助celery实现分布式爬虫, git repo: <https://git.pcl.ac.cn/huangwk/pano_distributed_crawler>

```
# physical machine
cd /home/pcl/traffic/pano_distributed_crawler
cd ./src; nohup sh start_panos.sh &
```

### IP资源池

简易高效的代理池

- API:
<http://localhost:5555/random> 获取一个随机可用代理

- 启动

```
# phisical mechine 
cd /home/pcl/traffic/ProxyPool
docker-compose up
```
