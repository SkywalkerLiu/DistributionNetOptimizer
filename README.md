# DistributionNetOptimizer

面向配电网规划前期场景构建与方案优化的 Python 项目。当前版本已经支持：

- 随机生成三维地形 `DTM`
- 生成地形派生栅格：坡度、坡向、粗糙度、可建设区、基础成本栅格
- 随机生成用户、树林、水区、人工禁区
- 将结果统一写入 `GeoPackage`
- 基于三维地形执行配变选址、放射式布线、ABC 相别分配、压降与净空校核
- 输出 2D 静态图、3D 静态图、3D 动态交互图

当前优化器遵循的核心约束：

- 高压侧为 `10kV` 放射式线路
- 低压侧为 `380/220V` 径向网络
- 树林、水区、人工禁区均不可穿越
- 用户只能作为终端，不能作为中间连接点
- 相邻杆塔平面档距不超过 `50m`
- 接户线平面档距不超过 `25m`
- 主干线路 `A/B/C/N` 同路径敷设
- 单相用户由优化算法自动分配到 `A/B/C`
- 线路沿全线必须满足最小离地净空，不足时会尝试自动增设杆塔修复

## 1. 安装

```bash
python -m pip install -r requirements.txt
```

依赖见 [requirements.txt](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/requirements.txt)：

- `numpy`
- `scipy`
- `rasterio`
- `geopandas`
- `shapely`
- `pyproj`
- `matplotlib`
- `plotly`
- `pyyaml`
- `networkx`
- `pytest`

## 2. 目录结构

```text
configs/
  default_config.yaml
  demo_small.yaml
data/
  terrain/
  masks/
  vector/
  outputs/
src/
  features/
  io/
  planning/
  terrain/
  viz/
tests/
PLAN_geo.md
PLAN_opt.md
```

## 3. 最常用命令

### 3.1 一键生成基础场景

```bash
python -m src.main generate-scene --config configs/default_config.yaml
```

小场景调试建议先跑：

```bash
python -m src.main generate-scene --config configs/demo_small.yaml
```

这个命令会完成：

- 地形生成
- 用户生成
- 障碍物生成
- 禁区掩膜生成
- 派生地形栅格生成
- 2D 场景图生成
- 3D 地形预览生成

### 3.2 只重算地形派生结果

```bash
python -m src.main derive-terrain --config configs/default_config.yaml
```

适用于已经修改了：

- `data/vector/features.gpkg`
- 或已有 `dtm.tif`

但只想重建：

- `forbidden_mask.tif`
- `buildable_mask.tif`
- `cost_base.tif`
- 坡度、坡向、粗糙度等派生栅格

### 3.3 只重绘 2D 场景图

```bash
python -m src.main plot-scene --config configs/default_config.yaml
```

### 3.4 只生成 3D 地形预览

```bash
python -m src.main plot-terrain-3d --config configs/default_config.yaml
```

输出：

- `data/outputs/plots/terrain_3d_preview.png`
- `data/outputs/plots/terrain_3d_preview.html`

### 3.5 导入或刷新人工禁区

```bash
python -m src.main refresh-manual --config configs/default_config.yaml --manual-geojson path/to/manual_constraints.geojson
```

如果你是直接在 `features.gpkg` 里修改了 `manual_no_build` 图层，也可以直接运行：

```bash
python -m src.main refresh-manual --config configs/default_config.yaml
```

### 3.6 执行优化规划

```bash
python -m src.main optimize-plan --config configs/default_config.yaml
```

这个命令会读取现有场景结果，并生成：

- 配变点
- 杆塔点
- 高压线路
- 低压主干线路
- 接户线
- ABC 相别分配结果
- 压降与净空校核结果
- 2D/3D 优化结果图
- 优化汇总 JSON

### 3.7 只重绘优化结果图

```bash
python -m src.main plot-plan --config configs/default_config.yaml
```

## 4. 当前默认用户模型

当前默认配置中，用户负荷由 `load_groups` 控制。

默认大场景 [configs/default_config.yaml](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/configs/default_config.yaml:1)：

- `40` 户默认 `7kW`
- `10` 户默认 `12kW`
- 功率因数统一为 `0.85`
- 默认均为单相用户

用户生成后会写入这些关键字段：

- `load_kw`
- `power_factor`
- `phase_type`
- `assigned_phase`
- `apparent_kva`
- `elev_m`

说明：

- `assigned_phase` 在生成场景时为空
- `optimize-plan` 执行后由优化器写入 `A/B/C`

## 5. 当前优化器能力

优化器实现位于 [src/planning/optimizer.py](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/src/planning/optimizer.py:1)。

当前版本已经包含：

- 三维配变选址
- 高压侧放射式路径搜索
- 低压侧径向树生成
- 用户只作为终端接入
- 动态杆塔布置
- 三相负荷平衡优化
- 下游线路分段负荷汇总
- 零线电流估算
- 电压降估算
- 导线全线净空检查
- 净空不足时自动加杆修复

### 5.1 三相分配

单相用户通过优化算法分配到 `A/B/C`，目标是在满足成本和电气约束的前提下，使：

- 台区总三相负荷尽量平衡
- 各线路分段三相负荷尽量平衡

### 5.2 三维净空约束

每条线路段都会沿线采样地形，检查导线与地表的最小净空：

- `hv_line` 使用高压净空阈值
- `lv_line` 使用低压净空阈值
- `service_drop` 使用接户线净空阈值

如果净空不足，优化器会尝试：

1. 在问题位置附近搜索可建设点
2. 自动加设修复杆塔
3. 将原线段拆分为多段重新校核

修复失败时，该段会被标记为违规，并在汇总结果中体现。

### 5.3 高压与低压分层

当前已区分 `10kV` 与 `380V`：

- 高压线路 `hv_line`
- 低压主干 `lv_line`
- 接户线 `service_drop`
- 高压杆 `hv_pole`
- 低压杆 `lv_pole`
- 高低压共杆 `hv_lv_shared`
- 净空修复杆 `clearance_repair`

配置中可分别设置：

- `hv_pole_height_m`
- `lv_pole_height_m`
- `transformer_hv_connection_height_m`
- `transformer_lv_connection_height_m`
- `hv_ground_clearance_m`
- `lv_ground_clearance_m`
- `service_ground_clearance_m`

## 6. 输出结果

稳定输出路径在 [src/main.py](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/src/main.py:432) 的 `resolve_paths()` 中定义。

主要输出包括：

- `data/terrain/dtm.tif`
- `data/terrain/slope.tif`
- `data/terrain/aspect.tif`
- `data/terrain/roughness.tif`
- `data/terrain/cost_base.tif`
- `data/masks/forbidden_mask.tif`
- `data/masks/buildable_mask.tif`
- `data/vector/features.gpkg`
- `data/outputs/plans/terrain_stats.json`
- `data/outputs/plans/optimization_summary.json`
- `data/outputs/plots/terrain_preview.png`
- `data/outputs/plots/slope_preview.png`
- `data/outputs/plots/scene_overview.png`
- `data/outputs/plots/terrain_with_features.png`
- `data/outputs/plots/forbidden_mask.png`
- `data/outputs/plots/terrain_3d_preview.png`
- `data/outputs/plots/terrain_3d_preview.html`
- `data/outputs/plots/optimized_plan_2d.png`
- `data/outputs/plots/optimized_plan_3d_static.png`
- `data/outputs/plots/optimized_plan_3d_dynamic.html`

## 7. GeoPackage 图层

`features.gpkg` 当前使用统一 schema，定义见 [src/io/vector_io.py](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/src/io/vector_io.py:8)。

主要图层：

- `users`
- `forest`
- `water`
- `manual_no_build`
- `candidate_transformer`
- `candidate_poles`
- `planned_lines`

### 7.1 `planned_lines` 关键字段

- `line_type`
- `from_node`
- `to_node`
- `phase_set`
- `service_phase`
- `horizontal_length_m`
- `length_3d_m`
- `dz_m`
- `slope_deg`
- `cost`
- `load_a_kva`
- `load_b_kva`
- `load_c_kva`
- `neutral_current_a`
- `voltage_drop_pct`
- `support_z_start_m`
- `support_z_end_m`
- `min_clearance_m`
- `required_clearance_m`
- `is_violation`

## 8. 可视化约定

### 8.1 相别颜色

当前统一采用：

- `A 相 = 黄`
- `B 相 = 绿`
- `C 相 = 红`

### 8.2 线路颜色

- `hv_line`：红褐色
- `lv_line`：深灰/黑色
- `service_drop`：橙色

### 8.3 杆塔颜色

- `hv_pole`：紫色
- `lv_pole`：青色
- `hv_lv_shared`：金黄色
- `clearance_repair`：粉色

### 8.4 图形输出

优化后会输出：

- `optimized_plan_2d.png`
- `optimized_plan_3d_static.png`
- `optimized_plan_3d_dynamic.html`

其中：

- 2D 图显示禁区、用户、杆塔、配变和线路
- 3D 静态图显示真实高程上的线路和支撑点
- 3D 动态图可旋转、缩放、查看 hover 信息

## 9. 配置文件说明

推荐先参考：

- [configs/default_config.yaml](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/configs/default_config.yaml:1)
- [configs/demo_small.yaml](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/configs/demo_small.yaml:1)

重点配置段：

- `scene`
- `terrain`
- `users`
- `obstacles`
- `planning`
- `outputs`
- `visualization`

### 9.1 `planning` 里常用的参数

- `transformer_candidate_step_m`
- `path_search_step_m`
- `transformer_capacity_kva`
- `max_loading_ratio`
- `transformer_fixed_cost`
- `pole_fixed_cost`
- `hv_pole_height_m`
- `lv_pole_height_m`
- `transformer_hv_connection_height_m`
- `transformer_lv_connection_height_m`
- `source_connection_height_m`
- `user_attachment_height_m`
- `line_cost_per_m`
- `service_line_cost_per_m`
- `hv_line_cost_per_m`
- `max_pole_span_m`
- `max_service_drop_m`
- `source_point_xy`
- `phase_balance_target_ratio`
- `phase_balance_max_ratio`
- `voltage_drop_max_pct`
- `hv_ground_clearance_m`
- `lv_ground_clearance_m`
- `service_ground_clearance_m`
- `clearance_sample_step_m`
- `clearance_search_radius_m`
- `clearance_max_repair_depth`

## 10. 推荐使用流程

### 10.1 先跑小场景

```bash
python -m src.main generate-scene --config configs/demo_small.yaml
python -m src.main optimize-plan --config configs/demo_small.yaml
python -m pytest -q
```

### 10.2 正式场景

```bash
python -m src.main generate-scene --config configs/default_config.yaml
python -m src.main optimize-plan --config configs/default_config.yaml
```

### 10.3 手工调整禁区后刷新

```bash
python -m src.main refresh-manual --config configs/default_config.yaml
python -m src.main optimize-plan --config configs/default_config.yaml
python -m src.main plot-plan --config configs/default_config.yaml
```

## 11. 测试

```bash
python -m pytest -q
```

当前测试覆盖重点包括：

- 用户 `load_groups` 接口
- 优化器输出基础可行结果
- 用户相别分配
- 接户线长度约束
- 用户不作为主干连接点
- 三维长度大于等于平面长度
- 3D 图输出

## 12. 已知注意事项

### 12.1 Windows 下 GeoPackage 文件锁

如果 `data/vector/features.gpkg` 正在被 QGIS、ArcGIS、Python Notebook 或其他程序占用，以下命令可能失败：

- `generate-scene`
- `optimize-plan`
- 任何需要重写 `features.gpkg` 的命令

出现类似 “另一个程序正在使用此文件” 时，请先关闭占用该文件的程序后重试。

### 12.2 当前净空模型仍是工程近似

目前净空检查基于：

- 支撑点连接高度
- 线性插值导线高度
- 沿线地形采样

还没有加入：

- 真实弧垂力学模型
- 转角杆 / 耐张杆结构约束
- 更精细的导线型号与机械校核

因此当前版本适合作为规划与方案筛选工具，不直接替代正式施工设计。

## 13. 相关规划文档

- [PLAN_geo.md](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/PLAN_geo.md)
- [PLAN_opt.md](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/PLAN_opt.md)
