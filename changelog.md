# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
****

## [Unreleased]

- AddedOnRampEdge 车道线设置

****

## [1.1.03] - 2021-09-09

### Added

- `main.py`
  - the framework of the total project
- `pano_base.py`
  - `pano_base_main` the main function in the module
- `pano_predict.py`
  - 原`predict_lanes.py`重构
- `pano_topo.py`
  - `bfs`
    - visted (src, dst);
    - change `if nxt_pid not in df_topo.index:`
  - `bidirection_bfs`
    - `pid not in df_topo.index and pid not in df_topo_prev.index`, or -> and
  - `combine_rids`
    - add `UnionFind` to find the origin rid
    - change the format of return.
  - `get_trajectory_by_rid`
- `db_process.py` add `predicted_url`.
- `utils.azimuth_helper.py`
- `utils.douglasPeucker.py`
  - compress alg for line and points
- `utils.unionFind.py`
  - to check the start edge of trajectory

### Changed

- `DigraphOSM`
  - `self.df_nodes` add "pid"

## [1.1.02] - 2021-08-19

### Code refactoring

- DigraphOSM.py
- pano_base.py
- pano_img.py
- panos_traverse.py

## [1.1.01] - 2021-08-10

### Added

- Add `DigraphOSM` to get and preporcess OSM data, and save it to the DB.

## [1.0.05] - 2021-07-08

Sort out and reorganize the project.

### Added

- `Osm_NET`
  - [x] 针对某些道路具有反向的情况，需要后边再细化了
  - [x] 筛选识别错误的车道情况

- `MatchingPanos`
  - [x] 保存到文件中

## [1.0.04] - 2021-04-24

### Added

- 增加 src/model/lstr -> lstr_pred_by_pid
  通过`pid`来预测车道情况，并绘制在图片上
- `Osm_NET`
  - [x] get_roads_by_road_level, 获取某一道路等级的所有车道IDs
  - [x] orginize_roads_based_grade
  - [x] get_roads_by_road_level

- `Sumo_Net`
  - [x] `plot_edge`

### Changed

- add `lstr_pred_by_pid` in `src/model/lstr`
- `Osm_NET`
  - [x] `get_pids_by_rid`
    - 道路延伸拓展或者裁剪；
    - AddedRamp连续
    - 增加memo，记录曾经匹配的记录
- `get_road_changed_section` add status

## [1.0.03] - 2021-04-24

### Added

- `OSM_Net`新增类

### Changed

- `label_json_parser.py` 增加对负样本的支持
- 针对id有`AddedOffRampEdge`和`AddedOnRampEdge`的处理优化
  invalid literal for int() with base 10: '208128050#5-AddedOffRampNode'

## [1.0.02] - 2021-04-22

### Added

- `sumo_net` 转换成类
- `MatchingPanos` 将原来匹配的代码转换成类
  - `plot_matching`可视化
- `SUMO_LOG`-> 记录变换的流程日志
  - 详情
  - 匹配的每一步

### Changed

- `lane_change_process` 测试待通过，name_to_id['科苑北路']

## [1.0.01] - 2021-04-16

### Added

- `count_panos_num_by_area` 新增查询各个区域的pano数量分布
- `get_road_changed_section` 优化函数，增加第二层过滤，考虑变道的阈值（20m）
- `modify_road_shape` 打通整个SUMO路网生成的流程，并以科技中二路（208128052）测试通过
- `.gitignore`  

### Changed

- modify_road_shape， 修复跨区间的情况
  
  如道路208128052，origin: [30, 33], insert: [24, 32] 跨区间怎么处理？

----
# history记录

- `draw_pred_lanes_on_img`:
  将pred结果绘制在图片上
- `pred_osm_road_by_rid`:
  预测OSM某一特定rid的道路的车道线，并按照特定的格式输出，如组合成一张照片，或者其他的组合

  ```
  # /home/pcl/traffic/RoadNetworkCreator_by_View/src/predict_lanes.py
  # 获取某一道路反方向的大图
  tmp = _get_revert_df_edges(-208128052, df_edges)
  pred_osm_road_by_rid(-208128052, tmp, True)
  pred_osm_road_by_rid(208128052, df_edges, True)
  ```

- `traverse_panos_by_road_name_new`
  以道路为到单位，遍历街景情况

  ```
  traverse_panos_by_road_name_new('打石一路', save_db=False)
  ```

## Develope Log

### First Version

- [X] Map visualization with satelite image

- [X] Split into module: Baidu View related API, RoadNetwork analysis
- [X] yaml文件的使用
- [X] Traverse the road. How to deal with the road with two directions
  - 将API反馈的路段逐一合并，巧妙利用端点的情况
- [X] Image storage format: file or DB
- [X] 等距离抽点：
- [X] 引入*args, **kwargs参数控制函数参数的输入
- [X] OSM dolwloader, 2020年12月3日

### 2020年12月4日

主要实现了区域的遍历、可视化、基础数据的抓取

- [X] baidu瓦片编号解析
- [X] BFS遍历方式
- [X] bfs 增加遍历的区域
- [X] links 绘制增加颜色，参考 <https://www.cnblogs.com/feffery/p/12361421.html>
- [X] 数据去重
- [X] 已存在的数据，避免重复访问 <- query_pano逻辑不够严谨，范围的不是道路终点的情况，把if改为while
    con = DB_pano_base.Links.apply(lambda x:  not isinstance(x, str ))
    con.sum()
- [X] 增加数据库备份功能backup

### 2020年12月9号

目标

- 实现`车道数量识别`
- 以光侨路为例，构建`车道级别的仿真路网`demo
- 交叉口识别 - 聚类算法
- 道路标志识别

traverse the topo network obtained from Baidu View

- [X] 道路遍历代码修改，以每条线的起点为端点开始遍历
- [X] get_road_shp_by_search_API增加缓存机制 -> `home\pcl\Data\minio_server\input\road_memo.csv`
- [X] 霍夫变换 测试， 效果不佳
- [X] GO语言重新编译
- [X] 标注工具使用和辅助代码开发:
  - `/home/pcl/traffic/RoadNetworkCreator_by_View/tools/labels/label_json_parser.py`
- [X] 图片数据裁剪
  - `/home/pcl/traffic/RoadNetworkCreator_by_View/tools/labels/label_helper.py`
- [X] 道路相似度检测，考虑因素：`线形`和`角度
- [X] Frenchet算法
- [X] 重新采集图片,然后统一resize大小到`(1280, 720)`
  
### 2021年1月20日

- [X] link筛选 -> 增加属性

- [X] `get_road_shp_by_search_API`增加城市信息，减少搜索范围
- [X] 增加位置信息`img_process.py`
- [X] 按照道路匹配pano并开始采集数据`road_matching.py`
- [X] 获取某个区域的道路名称 `get_roads_name_by_city`
- [X] 将原有代码模块化划分
- [X] 树上最长路径 [REF](https://www.lintcode.com/problem/longest-path-on-the-tree/description)
- [X] 裁剪图片-> TuSimple格式： `crop_img_for_lable`, 该函数也有直接复制的功能，不进行裁剪
- [X] pano_base代码增加`get_road_origin_points`， 获取道路的起始点，而不是针对每一个节点
- [X] 区域级别的街景遍历 `raverse_panos_in_district_level`

### 2021年1月27日

- [ ]

- [x] 增加关于dict，若pano的照片在其他路段出现过的时候，可以选择忽略跳过
- [x] 双向道路移动
- [x] 考虑融合`百度`和`OSM`路网数据
- [ ] 通过街景的情况识别`交叉口`，下一步识别是否为信号灯控制交叉口
- [ ] 实现加速遍历, 想法是两组的点坐标进行空间匹配
- [ ] 最后生成车道级别的路网，参考 AnyLogic`Converting GIS shapefile to a road network`

# 数据说明

## 研究区域

```
    bbox=[113.92348,22.57034, 113.94372,22.5855] # 留仙洞区域
    bbox=[113.92389,22.54080, 113.95558,22.55791] # 科技园中片区
    bbox = [114.04133,22.52903, 114.0645,22.55213] # 福田核心城区
```
