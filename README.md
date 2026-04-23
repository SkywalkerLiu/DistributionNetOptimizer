# DistributionNetOptimizer

这是一个按配置文件驱动的 Python 工程，用于生成台区配电线路优化前期所需的基础场景数据。当前版本已经实现：

- 随机生成 `DTM GeoTIFF` 地形底图
- 生成地形派生栅格：坡度、坡向、粗糙度、可建区、基础代价栅格
- 随机生成用户点、树林、水区、人工禁建区
- 将对象统一写入 `GeoPackage`
- 输出二维与三维预览结果，便于人工检查

## 1. 安装

```bash
python -m pip install -r requirements.txt
```

## 2. 项目结构

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
  io/
  terrain/
  features/
  planning/
  viz/
tests/
```

## 3. 最常用命令

### 3.1 一键生成完整场景

```bash
python -m src.main generate-scene --config configs/default_config.yaml
```

这个命令会一次性完成：

- 随机地形生成
- 用户点生成
- 障碍物生成
- 禁建掩膜生成
- 派生地形栅格生成
- 二维预览图生成
- 三维预览图生成

如果只是先做一个轻量验证，建议先用小场景配置：

```bash
python -m src.main generate-scene --config configs/demo_small.yaml
```

### 3.2 只重算地形派生结果

```bash
python -m src.main derive-terrain --config configs/default_config.yaml
```

适用于以下情况：

- 你已经修改了 `data/vector/features.gpkg`
- 你只想根据现有 `dtm.tif` 和矢量图层重新生成掩膜、代价栅格

### 3.3 只重画二维预览图

```bash
python -m src.main plot-scene --config configs/default_config.yaml
```

### 3.4 单独生成三维地形预览

```bash
python -m src.main plot-terrain-3d --config configs/default_config.yaml
```

这个命令会输出：

- `data/outputs/plots/terrain_3d_preview.png`
- `data/outputs/plots/terrain_3d_preview.html`

其中：

- `terrain_3d_preview.png` 是静态 3D 预览图
- `terrain_3d_preview.html` 是可交互的 3D 预览，可旋转、缩放、查看高程
- 三维预览会叠加 `users`、`forest`、`water`、`manual_no_build` 以及已有的规划线

注意：`generate-scene` 命令已经自动包含三维预览生成，此命令仅用于单独重新生成三维预览。

### 3.5 刷新 manual no-build

导入一个你手工画好的外部 `GeoJSON`，并自动刷新下游结果：

```bash
python -m src.main refresh-manual --config configs/default_config.yaml --manual-geojson path/to/manual_constraints.geojson
```

如果你不是导入外部 `GeoJSON`，而是直接在 `data/vector/features.gpkg` 里修改了 `manual_no_build` 图层，也可以直接运行：

```bash
python -m src.main refresh-manual --config configs/default_config.yaml
```

这个命令会自动完成：

- 合并外部 `GeoJSON` 到 `manual_no_build` 图层
- 重建 `data/masks/forbidden_mask.tif`
- 重建 `buildable_mask.tif`、`cost_base.tif` 等派生成果
- 重新输出二维和三维预览图片

## 4. 生成结果在哪里

默认输出目录如下：

- `data/terrain/dtm.tif`
- `data/terrain/slope.tif`
- `data/terrain/aspect.tif`
- `data/terrain/roughness.tif`
- `data/terrain/cost_base.tif`
- `data/masks/forbidden_mask.tif`
- `data/masks/buildable_mask.tif`
- `data/vector/features.gpkg`
- `data/outputs/plots/terrain_preview.png`
- `data/outputs/plots/slope_preview.png`
- `data/outputs/plots/scene_overview.png`
- `data/outputs/plots/terrain_with_features.png`
- `data/outputs/plots/forbidden_mask.png`
- `data/outputs/plots/terrain_3d_preview.png`
- `data/outputs/plots/terrain_3d_preview.html`
- `data/outputs/plans/terrain_stats.json`

`features.gpkg` 至少包含以下图层：

- `users`
- `forest`
- `water`
- `manual_no_build`
- `planned_lines`

## 5. 如何生成随机地形

随机地形由 `generate-scene` 命令触发，实际参数来自配置文件中的 `scene` 和 `terrain`。

关键参数在 [configs/default_config.yaml](configs/default_config.yaml)：

```yaml
scene:
  width_m: 5000
  height_m: 5000
  max_elevation_m: 500
  resolution_m: 1
  seed: 42

terrain:
  base_type: saddle
  add_perlin_noise: true
  noise_scale: 0.01
  noise_amplitude: 25.0
  noise_octaves: 4
  add_gaussian_hills: true
  hill_count: 12
  valley_ratio: 0.25
  smooth_sigma: 3.0
  clip_min: 0
  clip_max: 500
```

这些参数的作用可以这样理解：

- `seed`：控制随机性。相同 `seed` 会生成相同场景。
- `base_type`：基础地形趋势，目前支持 `saddle`、`ridge`、`basin` 这类类型。
- `add_perlin_noise`、`noise_scale`、`noise_amplitude`、`noise_octaves`：控制地形噪声起伏。
- `add_gaussian_hills`、`hill_count`、`valley_ratio`：控制局部丘陵和洼地数量。
- `smooth_sigma`：平滑程度，越大越圆润。
- `clip_min`、`clip_max`：最终高程范围裁剪。

最常见的修改方式：

1. 想换一张不同但可复现的随机地形：只改 `seed`
2. 想让地形更起伏：增大 `noise_amplitude` 或 `hill_count`
3. 想让地形更平滑：增大 `smooth_sigma`
4. 想缩小测试规模：改 `width_m`、`height_m`、`resolution_m`

## 6. 如何生成随机用户点

用户点由 `generate-scene` 自动生成，参数位于配置文件中的 `users`：

```yaml
users:
  count: 50
  min_spacing_m: 100
  distribution_mode: clustered
  cluster_count: 4
  cluster_radius_m: 300
  load_kw_range: [2.0, 25.0]
  importance_range: [1, 3]
```

含义如下：

- `count`：用户点数量
- `min_spacing_m`：用户点最小间距
- `distribution_mode`：分布方式，当前支持 `uniform` 和 `clustered`
- `cluster_count`：簇的数量
- `cluster_radius_m`：每个簇的半径范围
- `load_kw_range`：随机负荷范围
- `importance_range`：随机重要性等级范围

用户点生成时会自动满足：

- 不落在当前禁建区
- 不落在明显不可建区
- 自动从 `dtm.tif` 采样 `elev_m`

生成后的用户点写入：

- `data/vector/features.gpkg` 的 `users` 图层

## 7. 如何生成随机障碍物

随机障碍物也由 `generate-scene` 自动生成，参数位于配置文件中的 `obstacles`：

```yaml
obstacles:
  forest_count: 8
  water_count: 3
  manual_no_build_count: 2
  min_area_m2: 2000
  max_area_m2: 80000
  buffer_from_users_m: 20
```

含义如下：

- `forest_count`：树林数量
- `water_count`：水区数量
- `manual_no_build_count`：初始随机人工禁建区数量
- `min_area_m2`、`max_area_m2`：障碍物面积范围
- `buffer_from_users_m`：与用户点保持的安全缓冲距离

生成后的图层分别写入：

- `forest`
- `water`
- `manual_no_build`

并进一步生成：

- `data/masks/forbidden_mask.tif`

其中：

- `water` 默认视为禁行
- `manual_no_build` 默认视为禁建
- `forest` 既可能只是高通行成本区，也可能部分被标成禁行区

## 8. 可视化规范

### 8.1 统一颜色体系

| 要素 | 颜色 | 说明 |
|------|------|------|
| 地形低处 | 青 `#00ffff` | 最低海拔 |
| 地形中部 | 绿 `#00ff88` → 黄 `#ffff00` | 中等海拔 |
| 地形高处 | 橙 `#ff8800` → 红 `#ff0000` | 最高海拔 |
| 用户点 | 黑 `#000000` + 白边 | 小点 |
|  Forest | 浅绿 `#4a9e5c` + 深绿边 `#1d5e2e` | 半透明填充 |
| 水区 | 浅蓝 `#5ba8e8` + 深蓝边 `#1a4a8f` | 半透明填充 |
| 人工禁建区 | 无填充 + 红虚线边 `#e03c2a` | 虚线边框 |

### 8.2 要素显示规则

- 用户：统一用**黑色小点**（白色边框），不使用圆圈或其他符号
- 3D 图：用户点用小点，略微抬高避免被地形遮挡

### 8.3 图例规范

- 图例放在图的**右上角**
- 图例符号应与图中实际显示样式一致（干净示意符号）
- 图例背景半透明 (`framealpha=0.9`)

### 8.4 地形颜色渐变

- 2D/3D 统一采用 `cyan → green → yellow → orange → red` 渐变
- 低海拔：青/绿
- 中海拔：黄
- 高海拔：橙/红

## 9. 如何手动划取 `manual no-build` 区域

这里推荐两种方式。

### 方式 A：在 GIS 软件里画 GeoJSON，再导入

适合第一次补充禁建区，或者你更喜欢单独维护手工约束文件。

#### 步骤 1：先生成基础场景

```bash
python -m src.main generate-scene --config configs/default_config.yaml
```

#### 步骤 2：在 QGIS / ArcGIS 中手工画多边形

你可以新建一个 `Polygon` 图层，导出为 `GeoJSON`。默认配置的场景范围是：

- X 范围：`0 ~ 5000`
- Y 范围：`0 ~ 5000`
- CRS：`EPSG:3857`

如果你用的是 `demo_small.yaml`，则范围是：

- X 范围：`0 ~ 240`
- Y 范围：`0 ~ 240`
- CRS：`EPSG:3857`

建议字段如下：

- `obs_id`
- `source`
- `reason`
- `forbidden`

但这些字段不是必须的，缺失时程序会自动补默认值。

最小可用 `GeoJSON` 示例：

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "source": "manual",
        "reason": "school_buffer",
        "forbidden": 1
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [1000, 1000],
            [1400, 1000],
            [1400, 1400],
            [1000, 1400],
            [1000, 1000]
          ]
        ]
      }
    }
  ]
}
```

#### 步骤 3：导入并刷新

```bash
python -m src.main refresh-manual --config configs/default_config.yaml --manual-geojson path/to/manual_constraints.geojson
```

导入后会自动：

- 合并到 `features.gpkg` 的 `manual_no_build` 图层
- 重建禁建掩膜
- 重建派生栅格和候选点
- 重画二维图片

### 方式 B：直接编辑 `features.gpkg` 里的 `manual_no_build`

适合你已经在 GIS 软件中打开了整个场景工程。

#### 步骤 1：打开这个文件

- [data/vector/features.gpkg](data/vector/features.gpkg)

#### 步骤 2：直接编辑 `manual_no_build` 图层

你可以：

- 新增多边形
- 删除多边形
- 修改 `source`、`reason`、`forbidden`

建议保持 `forbidden = 1`。

#### 步骤 3：刷新下游成果

```bash
python -m src.main refresh-manual --config configs/default_config.yaml
```

如果你只想手工分步执行，也可以：

```bash
python -m src.main derive-terrain --config configs/default_config.yaml
python -m src.main plot-scene --config configs/default_config.yaml
```

## 10. 三维预览参数

三维预览使用配置文件中的 `visualization` 段：

```yaml
visualization:
  terrain_3d_max_grid_size: 200
  terrain_3d_vertical_exaggeration: 3.0
  terrain_3d_camera_elev_deg: 40
  terrain_3d_camera_azim_deg: -60
```

参数含义如下：

- `terrain_3d_max_grid_size`：3D 预览前的最大降采样网格边长
- `terrain_3d_vertical_exaggeration`：地形竖向夸张倍数
- `terrain_3d_camera_elev_deg`：相机俯视仰角
- `terrain_3d_camera_azim_deg`：相机方位角

为了避免在 `5000 x 5000` 正式场景上直接做原尺寸 3D 绘图，程序会先对 `dtm.tif` 做降采样再预览。

## 11. 推荐的实际使用流程

### 轻量测试

```bash
python -m src.main generate-scene --config configs/demo_small.yaml
python -m pytest -q
```

### 正式生成一版 3km × 3km 场景

```bash
python -m src.main generate-scene --config configs/default_config.yaml
```

### 生成后人工补画禁建区

```bash
python -m src.main refresh-manual --config configs/default_config.yaml --manual-geojson path/to/manual_constraints.geojson
```

### 只调整随机参数并重新生成

修改 [configs/default_config.yaml](configs/default_config.yaml) 里的：

- `scene.seed`
- `terrain.*`
- `users.*`
- `obstacles.*`
- `visualization.*`

然后重新执行：

```bash
python -m src.main generate-scene --config configs/default_config.yaml
```

## 12. 测试

```bash
python -m pytest -q
```

当前测试覆盖：

- GeoTIFF 读写
- GeoPackage 图层读写
- 地形生成尺寸与范围
- 随机种子可复现
- 用户点与障碍物的空间一致性
- 3D 地形预览文件输出

## 13. 当前实现说明

目前版本已经能作为首版场景生成器使用，但有几点需要知道：

- `generate-scene` 会重写当前输出文件
- `manual_no_build` 支持随机初始生成，也支持后续手工补画
- `plot-terrain-3d` 依赖已有的 `dtm.tif`
- `planned_lines` 是为后续优化阶段预留的接口层
- 默认场景 `3000 x 3000 @ 1m` 数据量适中，第一次建议先用 `demo_small.yaml` 验证流程
