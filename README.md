# DistributionNetOptimizer

DistributionNetOptimizer 用于生成配电网规划前期的地形场景，并在此基础上进行低压配电网络优化。当前默认场景聚焦农村台区形态：居民小簇为主，少量散户分布在簇外。

## 核心能力

- 生成 DTM、坡度、粗糙度、可建设区与禁建区栅格。
- 生成用户、树林、水域、人工禁建区等场景图层。
- 支持“居民成簇 + 少量散户”的用户生成模式。
- 输出场景 2D 图、3D 静态地形图和 3D 动态地形 HTML。
- 将场景预览与优化结果预览解耦，避免地形预览混入线路、杆塔、配变。
- 优化阶段支持配变、杆塔、低压线路、接户线和用户接入结果展示。

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

用户总数固定为 50 户：

- 簇内居民：45 户。
- 散户：5 户。
- 每个居民簇 3-8 户。
- 簇内任意两户距离不超过 10m。
- 散户之间至少 30m。
- 散户与任一簇中心至少 30m。

负荷配置：

- 普通住户：40 户，7.0kW，功率因数 0.85，单相。
- 大功率用户：10 户，10.0kW，功率因数 0.85，单相。

`users` 图层包含 `user_type`、`settlement_type`、`cluster_id` 等字段，用于区分普通住户/大功率用户、簇内居民/散户和簇编号。

## 常用命令

安装依赖：

```bash
python -m pip install -r requirements.txt
```

生成地形、用户和场景图层：

```bash
python -m src.main generate-scene --config configs/default_config.yaml
```

重新绘制 2D 场景图：

```bash
python -m src.main plot-scene --config configs/default_config.yaml
```

重新绘制 3D 地形预览：

```bash
python -m src.main plot-terrain-3d --config configs/default_config.yaml
```

运行优化：

```bash
python -m src.main optimize-plan --config configs/default_config.yaml
```

重新绘制优化结果：

```bash
python -m src.main plot-plan --config configs/default_config.yaml
```

## 输出文件

场景生成结果：

```text
data/outputs/plots/scene_2d.png
data/outputs/plots/terrain_3d_preview.png
data/outputs/plots/terrain_3d_preview.html
```

优化规划结果：

```text
data/outputs/plots/optimized_plan_2d.png
data/outputs/plots/optimized_plan_3d_static.png
data/outputs/plots/optimized_plan_3d_dynamic.html
```

`terrain_3d_preview.html` 只展示地形生成结果：

- 地形表面
- 用户点
- 树林
- 水域
- 人工禁建区

它不展示：

- 优化线路
- 杆塔
- 配变
- 接户线

`optimized_plan_3d_dynamic.html` 才展示优化后的线路、杆塔、配变和用户接入结果。

## 3D 地形悬停信息

在 `terrain_3d_preview.html` 中，鼠标悬停用户点可查看：

- 用户编号，例如 `U001`
- 用户类型，例如普通住户或大功率用户
- 聚落类型，例如簇内居民或散户
- 簇编号，例如 `C001` 或 `-`
- 负荷，例如 `7.0 kW`
- 相别类型，例如 `single`

## 主要代码位置

- [src/features/users_generator.py](src/features/users_generator.py)：用户生成逻辑。
- [src/io/vector_io.py](src/io/vector_io.py)：GeoPackage 图层 schema。
- [src/viz/plot_scene.py](src/viz/plot_scene.py)：2D 场景图。
- [src/viz/plot_terrain_3d.py](src/viz/plot_terrain_3d.py)：3D 地形预览和优化结果 3D 输出。
- [src/viz/plot_optimized_plan.py](src/viz/plot_optimized_plan.py)：优化结果图输出。
- [src/main.py](src/main.py)：CLI 入口。

## 测试

运行全量测试：

```bash
python -m pytest -q
```

当前重点覆盖：

- `clustered_with_scattered` 用户数量与散户数量。
- 每簇 3-8 户。
- 簇内任意两户距离不超过 10m。
- 40 户普通住户、10 户大功率用户。
- 所有用户功率因数为 0.85。
- 所有用户均为单相。
- `terrain_3d_preview.html` 不包含优化图层。
- `optimized_plan_3d_dynamic.html` 包含优化图层。
- 3D 地形用户悬停信息包含用户编号和负荷。

## 说明

当前优化器是规划级近似求解器，不是施工设计级详设工具。地形场景预览用于检查生成结果，优化结果预览用于检查规划后的线路、杆塔、配变和接入关系，两者输出路径和绘图逻辑已经分离。
