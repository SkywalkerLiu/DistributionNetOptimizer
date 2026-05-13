# DistributionNetOptimizer

DistributionNetOptimizer 用于生成配电网规划前期的地形场景，并在此基础上进行低压配电网优化。项目保留默认随机仿真场景，同时新增基于奥维截图人工整理参数的确定性台区场景。

默认场景为 `400m x 600m`、50 户用户，支持居民簇与散户混合分布、地形约束、禁建区、三相分配、电压评估、杆塔派生和推荐配变容量。

奥维截图场景使用 `configs/ovi_screenshot_config.yaml`，范围为 `428.45m x 909.30m`、1m 分辨率、25 个固定用户点，以及确定性的水塘、林地和禁建区 polygon。该模式不做运行时图片识别，也不会在配置错误时 fallback 到随机场景。

## 场景配置

默认随机场景：

```bash
python -m src.main generate-scene --config configs/default_config.yaml
python -m src.main derive-terrain --config configs/default_config.yaml
python -m src.main optimize-plan --config configs/default_config.yaml
python -m src.main plot-plan --config configs/default_config.yaml
```

奥维截图确定性场景：

```bash
python -m src.main generate-scene --config configs/ovi_screenshot_config.yaml
python -m src.main derive-terrain --config configs/ovi_screenshot_config.yaml
python -m src.main optimize-plan --config configs/ovi_screenshot_config.yaml
python -m src.main plot-plan --config configs/ovi_screenshot_config.yaml
```

奥维场景关键配置：

```yaml
scene:
  width_m: 428.45
  height_m: 909.30
  resolution_m: 1
  origin_x_m: 0
  origin_y_m: 909.30
  crs: EPSG:3857

terrain:
  base_type: ovi_screenshot
  clip_min: 220
  clip_max: 420
  max_buildable_slope_deg: 35.0

users:
  source: ovi_screenshot
  count: 25

obstacles:
  source: ovi_screenshot
```

## Ovi Screenshot 模式

`ovi_screenshot` 模式只替换 `generate-scene` 阶段的数据来源：

- `src/terrain/ovi_screenshot_terrain.py` 生成确定性 DTM。
- `src/features/ovi_screenshot_features.py` 生成固定用户点、水塘、林地和禁建区。
- `terrain_generator.py`、`users_generator.py`、`obstacles_generator.py` 根据配置选择新分支。

该模式保持 `features.gpkg` 标准图层和字段 schema 不变：

```text
users
forest
water
manual_no_build
candidate_transformer
candidate_poles
planned_lines
```

奥维原图可放在：

```text
data/raw/ovi_taizone_original.png
```

图片仅作为人工复核和预览背景来源，不参与运行时识别流程。

## V2 优化器

V2 优化器采用 `voltage_first` 规划逻辑：候选方案排序优先比较用户整体压降，其次比较建设成本，再比较线损和三相平衡。

链路如下：

```text
候选走廊图 -> 自动 service group 识别 -> 共享接户杆候选生成 -> 径向树选边 -> 三相分配 -> 功率流评估 -> 电压优先排序 -> 杆塔派生与几何复核 -> 推荐配变容量
```

`service group` 根据用户坐标、接户线长度、杆位可建设性和净距约束自动生成，不依赖模拟数据中的 `cluster_id` 或 `settlement_type`。`cluster_id` 仅用于场景生成阶段的展示和调试，优化器会重新写入 `service_group_id`、`service_group_size` 和 `is_service_singleton`。

配变容量默认不在优化前固定。算法先完成网络规划，再根据最终接入负荷、需用系数和目标负载率，从标准容量序列中推荐配变容量。

```yaml
planning:
  transformer_capacity_kva: null
  transformer_capacity_enforced: false
  demand_factor: 0.85
  transformer_target_loading_ratio: 0.80
```

## 常用命令

安装依赖：

```bash
python -m pip install -r requirements.txt
```

单独重绘场景或地形预览：

```bash
python -m src.main plot-scene --config configs/ovi_screenshot_config.yaml
python -m src.main plot-terrain-3d --config configs/ovi_screenshot_config.yaml
```

运行测试：

```bash
python -m pytest -q
```

只跑奥维场景相关测试：

```bash
python -m pytest -q tests/test_ovi_screenshot_scene.py
```

## 输出文件

地形和 mask：

```text
data/terrain/dtm.tif
data/terrain/slope.tif
data/terrain/aspect.tif
data/terrain/roughness.tif
data/terrain/cost_base.tif
data/masks/forbidden_mask.tif
data/masks/buildable_mask.tif
```

矢量图层：

```text
data/vector/features.gpkg
```

场景预览：

```text
data/outputs/plots/scene_2d.png
data/outputs/plots/terrain_3d_preview.png
data/outputs/plots/terrain_3d_preview.html
```

优化结果：

```text
data/outputs/plans/optimization_summary.json
data/outputs/plots/optimized_plan_2d.png
data/outputs/plots/optimized_plan_3d_static.png
data/outputs/plots/optimized_plan_3d_dynamic.html
```

## Summary 字段

`optimization_summary.json` 的 `transformer` 字段包含：

- `raw_loading_kva`
- `demand_factor`
- `effective_loading_kva`
- `target_loading_ratio`
- `required_capacity_kva`
- `recommended_capacity_kva`
- `recommended_loading_ratio`

`voltage` 字段包含：

- `mean_user_voltage_drop_pct`
- `load_weighted_mean_user_voltage_drop_pct`
- `p95_user_voltage_drop_pct`
- `max_total_voltage_drop_pct`

`service_grouping` 字段包含：

- `group_count`
- `multi_user_group_count`
- `singleton_group_count`
- `max_group_size`
- `shared_attach_group_count`
- `groups_with_multiple_attach_nodes`
- `max_attach_nodes_per_group`
- `extra_attach_penalty`

`path_constraints` 字段记录用户路径约束是否启用、实际最长路径、超限用户数量和最长路径用户列表。默认随机场景使用 `300m` 用户路径约束；奥维截图场景因台区更狭长，在配置中使用 `800m` 作为场景级适配。

常见不可行原因包括：

- `user_path_too_long`
- `service_drop_too_long`
- `radial_tree_infeasible`
- `line_vertical_clearance_exceeded`
- `line_user_clearance_exceeded`
- `pole_user_clearance_exceeded`

仅当 `transformer_capacity_enforced=true` 且配置了 `transformer_capacity_kva` 时，才会触发 `transformer_overloaded`。仅当 `phase_balance_hard_constraint=true` 时，三相不平衡才会触发 `phase_unbalance_exceeded`；默认情况下三相平衡作为软优化目标参与 penalty。

## 测试覆盖

当前重点覆盖：

- 默认随机场景仍可生成用户、障碍和地形。
- `ovi_screenshot` DTM 尺寸和高程趋势稳定。
- `ovi_screenshot` 用户点固定为 25 户。
- `ovi_screenshot` 水塘、林地、禁建区为确定性 polygon。
- `features.gpkg` 标准图层字段 schema 保持不变。
- V2 voltage-first 排序、service grouping、共享接户杆、配变容量推荐、路径诊断、线损和三相平衡输出。

## 说明

当前优化器是规划级近似求解器，不是施工设计级详设工具。地形场景预览用于检查生成结果，优化结果预览用于检查规划后的线路、杆塔、配变和接入关系。
