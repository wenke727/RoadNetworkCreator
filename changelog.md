# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
****

## [Unreleased]
- AddedOnRampEdge 车道线设置
****

## [1.0.04] - 2021-04-24

### Added

- `Osm_NET`
  - [x] get_roads_by_road_level, 获取某一道路等级的所有车道IDs
  - [ ] 针对某些道路具有反向的情况，需要后边再细化了
  - [ ] 筛选识别错误的车道情况
  - [x] orginize_roads_based_grade
  - [x] get_roads_by_road_level

- `Sumo_Net`
  - [x] `plot_edge`

- `MatchingPanos`
  - [ ] 保存到文件中

### Changed

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
