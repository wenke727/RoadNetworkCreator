# RoadNetworkCreator_by_View

## Develope Log
### First Version
- [X] Map visualization with satelite image
- [X] Split into module: Baidu View related API, RoadNetwork analysis
- [X] yaml文件的使用
- [X] Traverse the road. How to deal with the road with two directions
    * 将API反馈的路段逐一合并，巧妙利用端点的情况
- [X] Image storage format: file or DB
- [X] 等距离抽点：
- [X] 引入*args, **kwargs参数控制函数参数的输入
- [X] OSM dolwloader, 2020年12月3日

### 2020年12月4日
主要实现了区域的遍历、可视化、基础数据的抓取
- [X] baidu瓦片编号解析
- [X] BFS遍历方式
- [X] bfs 增加遍历的区域
- [X] links 绘制增加颜色，参考 https://www.cnblogs.com/feffery/p/12361421.html
- [X] 数据去重
- [X] 已存在的数据，避免重复访问 <- query_pano逻辑不够严谨，范围的不是道路终点的情况，把if改为while
    con = DB_pano_base.Links.apply(lambda x:  not isinstance(x, str ))
    con.sum()
- [X] 增加数据库备份功能backup


### 2020年12月9号
目标
* 实现`车道数量识别`
* 以光侨路为例，构建`车道级别的仿真路网`demo
* 交叉口识别 - 聚类算法
* 道路标志识别

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
- [ ] 增加关于dict，若pano的照片在其他路段出现过的时候，可以选择忽略跳过
- [ ] 双向道路移动
- [ ] 考虑融合`百度`和`OSM`路网数据
- [ ] 通过街景的情况识别`交叉口`，下一步识别是否为信号灯控制交叉口
- [ ] 实现加速遍历, 想法是两组的点坐标进行空间匹配
- [ ] 最后生成车道级别的路网，参考 AnyLogic`Converting GIS shapefile to a road network`


# 数据说明




