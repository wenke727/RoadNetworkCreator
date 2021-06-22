# RoadNetworkCreator_by_View

## API

### 视网膜系统

启动

```
# physical machine
cd /home/pcl/server/pcl/pano_data
nohup python3 pano.py > pano.log &
```

API情况

- 获取临近点:
  <http://192.168.135.15:4000/get_nearby_panos?lon=113.93412&lat=22.575369>

- pano_data:
  <http://192.168.135.15:4000/get_pano_data?feature=point&bbox=113.929807,22.573702,113.937680,22.578734>

### 车道线预测API

API

- <http://192.168.135.15:5000/lstr?fn=91868b-159f-3c50-008c-55ec99_04_01005700001311231236314705I_253.jpg>

  ```
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

## 分布式街景爬虫

借助celery实现分布式爬虫, git repo: <https://git.pcl.ac.cn/huangwk/pano_distributed_crawler>

```
# physical machine
cd /home/pcl/traffic/pano_distributed_crawler
cd ./src; nohup sh start_panos.sh &
```

## IP资源池

简易高效的代理池

API:

- <http://localhost:5555/random> 获取一个随机可用代理

启动

```
# phisical mechine 
cd /home/pcl/traffic/ProxyPool
docker-compose up
```
