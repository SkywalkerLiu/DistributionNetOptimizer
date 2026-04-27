# DistributionNetOptimizer

DistributionNetOptimizer 用于生成配电网规划前期的地形场景，并在此基础上进行低压配电网优化。默认场景为 400m x 600m、50 户用户，支持居民簇与散户混合分布、地形约束、禁建区、三相分配、电压评估、杆塔派生和推荐配变容量。

## V2 优化器

V2 优化器采用 voltage-first 规划逻辑：候选方案排序优先比较用户整体压降，其次比较建设成本，再比较线损和三相平衡。

架构链路：

```text
候选走廊图 -> 自动 service group 识别 -> 共享接户杆候选生成 -> 径向树选边 -> 三相分配 -> 功率流评估 -> 电压优先排序 -> 杆塔派生与几何复核 -> 推荐配变容量
```

V2 优化器会根据用户点空间分布自动识别 service group，用于判断哪些用户可以共用一根接户杆。该机制只依赖用户坐标、接户线长度、杆位可建设性和净距约束，不依赖模拟数据中的 `cluster_id` 或 `settlement_type`。因此，真实台区用户点数据无需预先标注 cluster，也可以自动生成共享接户杆方案。

每个 service group 生成共享接入候选点，散户作为单户 service group 处理。组内用户通过各自的 `service_drop` 接入同一根共享接户杆；若初始组过大或无法满足接户距离、净距和可建设性约束，优化器会在 service grouping 内部继续拆分，而不是回退到逐户 attach 建模。

注意：`cluster_id` 是随机场景生成阶段用于展示和调试的可选字段，不参与 V2 优化器的 service grouping 逻辑。优化器运行时会重新根据用户坐标生成 `service_group_id`。

## 默认配置

默认配置文件见 [configs/default_config.yaml](configs/default_config.yaml)。

```yaml
planning_v2:
  service_grouping_enabled: true
  service_group_neighbor_radius_m: 12.0
  service_group_min_users: 2
  service_group_max_users: 8
  service_group_max_diameter_m: 20.0
  service_group_max_service_drop_m: 25.0
  service_group_attach_search_radius_m: 35.0
  service_group_extra_attach_penalty: 200000.0
```

配变容量默认不在优化前固定。算法先完成网络规划，再根据最终接入负荷、需用系数和目标负载率，从标准容量序列中推荐配变容量。

```yaml
planning:
  transformer_capacity_kva: null
  transformer_capacity_enforced: false
  demand_factor: 0.85
  transformer_target_loading_ratio: 0.80
```

用户到配变的低压供电路径按“公共低压线路路径长度 + 接户线长度”计算：

```yaml
planning_v2:
  max_user_path_length_m: 300.0
  enforce_max_user_path_length: true
```

公共低压主干/分支线路默认采用 `lv_ground_clearance_m=6.0m`。接户线由 `service_ground_clearance_m` 单独控制。默认低压杆塔高度为 `lv_pole_height_m=10.0m`。

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
data/vector/features.gpkg
```

`optimization_summary.json` 中新增 `service_grouping` 字段，包含：

- `group_count`
- `multi_user_group_count`
- `singleton_group_count`
- `max_group_size`
- `shared_attach_group_count`
- `groups_with_multiple_attach_nodes`
- `max_attach_nodes_per_group`
- `extra_attach_penalty`

`users` 图层会输出 `service_group_id`、`service_group_size` 和 `is_service_singleton`。

`candidate_poles` 图层中，共享接户杆会标记为 `pole_type=shared_service_pole`，并输出 `service_group_id` 和 `served_user_count`。

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

轻量相关测试：

```bash
python -m pytest -q tests/test_service_grouping.py tests/test_optimizer.py
```

完整测试：

```bash
python -m pytest -q
```

当前重点覆盖：自动 service grouping、共享接户杆、V2 voltage-first 排序、配变容量后置推荐、300m 用户路径硬约束、6m 公共低压线路净空、10m 杆塔高度、线损输出和三相平衡软目标。

## 说明

当前优化器是规划级近似求解器，不是施工设计级详设工具。地形场景预览用于检查生成结果，优化结果预览用于检查规划后的线路、杆塔、配变和接入关系。
