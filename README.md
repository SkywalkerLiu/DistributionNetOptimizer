# DistributionNetOptimizer

DistributionNetOptimizer 用于生成配电网规划前期的地形场景，并在此基础上进行低压配电网优化。当前默认场景为 400m x 600m、50 户用户，采用 `clustered_with_scattered` 用户分布，并区分普通用户与大功率用户负荷。

## V2 优化器

当前 V2 优化器采用 voltage-first 规划逻辑：先基于候选走廊图构建径向低压网络，再进行三相分配和功率流评估。候选方案排序优先比较用户整体压降，其次比较建设成本，再比较线损和三相平衡。

架构链路：

```text
候选走廊图 -> 径向树选边 -> 三相分配 -> 功率流评估 -> 电压优先排序 -> 杆塔派生与几何复核 -> 推荐配变容量
```

V2 默认不在优化前固定配变容量。算法先完成网络规划，再根据最终接入负荷、需用系数和目标负载率，从标准容量序列中推荐配变容量。

```yaml
planning:
  transformer_capacity_kva: null
  transformer_capacity_enforced: false
  demand_factor: 0.85
  transformer_target_loading_ratio: 0.80
  transformer_standard_capacities_kva: [100, 160, 200, 250, 315, 400, 500, 630, 800]
```

用户到配变的低压供电路径按“公共低压线路路径长度 + 接户线长度”计算：

```yaml
planning_v2:
  max_user_path_length_m: 300.0
  enforce_max_user_path_length: true
```

当总路径长度超过 `max_user_path_length_m` 且 `enforce_max_user_path_length=true` 时，该方案判定为不可行，并输出 `user_path_too_long`。

公共低压主干/分支线路默认采用 `lv_ground_clearance_m=6.0m`。接户线仍由 `service_ground_clearance_m` 单独控制。默认低压杆塔高度为 `lv_pole_height_m=10.0m`。

## 默认场景

默认配置文件为 [configs/default_config.yaml](configs/default_config.yaml)。

```yaml
scene:
  width_m: 400
  height_m: 600
  max_elevation_m: 100
  resolution_m: 1
  origin_x_m: 0
  origin_y_m: 600
  crs: EPSG:3857
  seed: 66
```

用户配置：

- 普通住户：40 户，7.0 kW，功率因数 0.85，单相。
- 大功率用户：10 户，10.0 kW，功率因数 0.85，单相。
- 45 户位于居民簇内，5 户为散户。
- 每个居民簇 3-8 户，簇内直径不超过 10m。

## 常用命令

安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行完整流程：

```bash
python -m src.main generate-scene --config configs/default_config.yaml
python -m src.main derive-terrain --config configs/default_config.yaml
python -m src.main optimize-plan --config configs/default_config.yaml
python -m src.main plot-plan --config configs/default_config.yaml
```

单独重绘场景或预览：

```bash
python -m src.main plot-scene --config configs/default_config.yaml
python -m src.main plot-terrain-3d --config configs/default_config.yaml
```

## 输出文件

场景输出：

```text
data/outputs/plots/scene_2d.png
data/outputs/plots/terrain_3d_preview.png
data/outputs/plots/terrain_3d_preview.html
```

优化输出：

```text
data/outputs/plans/optimization_summary.json
data/outputs/plots/optimized_plan_2d.png
data/outputs/plots/optimized_plan_3d_static.png
data/outputs/plots/optimized_plan_3d_dynamic.html
```

`terrain_3d_preview.html` 只展示地形、用户、森林、水域和人工禁建区；`optimized_plan_3d_dynamic.html` 展示优化后的线路、杆塔、配变和用户接入结果。

## Summary 字段

`optimization_summary.json` 的 `transformer` 字段会输出：

- `raw_loading_kva`
- `demand_factor`
- `effective_loading_kva`
- `target_loading_ratio`
- `required_capacity_kva`
- `recommended_capacity_kva`
- `recommended_loading_ratio`

`voltage` 字段会输出：

- `mean_user_voltage_drop_pct`
- `load_weighted_mean_user_voltage_drop_pct`
- `p95_user_voltage_drop_pct`
- `max_total_voltage_drop_pct`

`path_constraints` 字段会输出 300m 路径约束是否启用、实际最大路径长度、超限用户数量和最长路径用户列表。

默认不可行原因包括：

- `user_path_too_long`
- `service_drop_too_long`
- `radial_tree_infeasible`
- `line_vertical_clearance_exceeded`
- `line_user_clearance_exceeded`
- `pole_user_clearance_exceeded`

仅当 `transformer_capacity_enforced=true` 且配置了 `transformer_capacity_kva` 时，才会触发 `transformer_overloaded`。仅当 `phase_balance_hard_constraint=true` 时，三相不平衡才会触发 `phase_unbalance_exceeded`；默认情况下三相平衡作为软优化目标参与 penalty。

## 测试

```bash
python -m pytest -q
```

当前重点覆盖：场景生成、V2 voltage-first 排序、配变容量后置推荐、300m 用户路径硬约束、6m 公共低压线路净空、10m 杆塔高度、线损输出和三相平衡软目标。

## 说明

当前优化器是规划级近似求解器，不是施工设计级详设工具。地形场景预览用于检查生成结果，优化结果预览用于检查规划后的线路、杆塔、配变和接入关系。
