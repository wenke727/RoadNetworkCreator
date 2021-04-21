# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
****
## [Unreleased]


****
## [1.0.01] - 2021-04-16
### Added
- `count_panos_num_by_area` 新增查询各个区域的pano数量分布
- `get_road_changed_section` 优化函数，增加第二层过滤，考虑变道的阈值（20m）
- `modify_road_shape` 打通整个SUMO路网生成的流程，并以科技中二路（208128052）测试通过
- `.gitignore`  

### Changed
- modify_road_shape， 修复跨区间的情况
  如道路208128052，origin: [30, 33], insert: [24, 32] 跨区间怎么处理？
****
## [1.0.0] - 2021-04-21
### Added

### Changed

### Removed

### Fixed

