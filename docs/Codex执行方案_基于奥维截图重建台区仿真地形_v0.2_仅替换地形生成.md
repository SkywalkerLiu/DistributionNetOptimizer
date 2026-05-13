# Codex 执行方案 v0.2：仅替换地形场景生成，保留后续优化规划链路

## 0. 本版方案的纠偏说明

上一版方案把项目改造范围扩大到了“完整重建一套场景与规划流程”，这会偏离当前项目。  
本版只针对 **地形场景生成阶段** 做改动：把原来的随机地形、随机用户、随机障碍，替换为基于奥维截图的确定性台区场景；后续 `derive-terrain`、`optimize-plan`、`plot-plan` 的主流程和 V2 优化器原则上保持不动。

本版目标是：

```text
奥维截图原型
  ↓
确定性仿真地形 DTM + 用户点 + 约束图层
  ↓
复用现有 terrain derivatives / cost surface
  ↓
复用现有 V2 优化器
  ↓
复用现有线路、杆塔、配变、电气校核和可视化输出
```

不要把本任务做成通用“图片识别系统”，也不要把后续优化器推倒重写。

---

## 1. 当前项目链路必须保留

当前项目已经具备完整链路：

```bash
python -m src.main generate-scene --config configs/default_config.yaml
python -m src.main derive-terrain --config configs/default_config.yaml
python -m src.main optimize-plan --config configs/default_config.yaml
python -m src.main plot-plan --config configs/default_config.yaml
```

本次改动后，新增一套奥维截图配置，推荐运行：

```bash
python -m src.main generate-scene --config configs/ovi_screenshot_config.yaml
python -m src.main derive-terrain --config configs/ovi_screenshot_config.yaml
python -m src.main optimize-plan --config configs/ovi_screenshot_config.yaml
python -m src.main plot-plan --config configs/ovi_screenshot_config.yaml
```

除非出现接口兼容问题，不要修改以下模块的核心算法逻辑：

```text
src/planning/optimizer_v2.py
src/planning/cost_surface.py
src/viz/plot_optimized_plan.py
src/terrain/terrain_derivatives.py
src/io/raster_io.py
src/io/vector_io.py
```

---

## 2. 本次改动范围

### 2.1 需要改动

只改动或新增以下内容：

```text
configs/ovi_screenshot_config.yaml              # 新增奥维截图场景配置
src/terrain/ovi_screenshot_terrain.py           # 新增确定性地形生成器
src/features/ovi_screenshot_features.py         # 新增确定性用户点、道路、障碍图层
src/terrain/terrain_generator.py                # 增加 ovi_screenshot 分支
src/features/users_generator.py                 # 增加 ovi_screenshot 分支
src/features/obstacles_generator.py             # 增加 ovi_screenshot 分支
tests/test_ovi_screenshot_scene.py              # 新增轻量测试
```

如果当前项目结构与上述文件名不完全一致，可以按现有结构合并，但必须保持职责清晰：

```text
terrain_generator 只负责 DTM；
users_generator 只负责用户点；
obstacles_generator 只负责 forest / water / manual_no_build；
main.generate_scene 只负责串联，不写具体地形公式。
```

### 2.2 不要改动

不要重写或大改以下内容：

```text
V2 优化器
service grouping 逻辑
共享接户杆逻辑
径向树选边逻辑
三相分配逻辑
压降、电气校核逻辑
杆塔派生逻辑
优化结果输出结构
features.gpkg 图层 schema
```

### 2.3 允许的配置调整

由于奥维截图台区范围约为 `428.45m × 909.30m`，比原默认 `400m × 600m` 更狭长。如果默认 `max_user_path_length_m=300m` 导致整体不可行，允许在 `configs/ovi_screenshot_config.yaml` 中做 **配置级调整**，例如：

```yaml
planning_v2:
  max_user_path_length_m: 500.0
  enforce_max_user_path_length: true
```

这属于适配新场景尺寸，不属于改动优化算法。

---

## 3. 奥维截图场景的固定几何参数

### 3.1 台区范围

根据图片红框与图中测距标注，本台区仿真范围采用局部平面坐标：

```text
左下角 = (0.00, 0.00)
右下角 = (428.45, 0.00)
左上角 = (0.00, 909.30)
右上角 = (428.45, 909.30)
```

配置应设置为：

```yaml
scene:
  width_m: 428.45
  height_m: 909.30
  resolution_m: 1.0
  origin_x_m: 0.0
  origin_y_m: 909.30
  crs: EPSG:3857
  seed: 66
```

说明：

1. 这里使用 `EPSG:3857` 只是为了沿用当前项目的米制平面坐标习惯；
2. 不声明其为真实测绘坐标；
3. 该坐标系只用于仿真、算法验证和可视化。

### 3.2 图片像素校准参数

Codex 可以把原图放在：

```text
data/raw/ovi_taizone_original.png
```

但运行时不要做复杂图片识别。图片只作为人工复核和预览背景。

红框像素校准可写入配置或常量：

```yaml
ovi_screenshot:
  image_path: data/raw/ovi_taizone_original.png
  roi_px:
    left: 681
    top: 218
    right: 1094
    bottom: 1091
  roi_size_m:
    width: 428.45
    height: 909.30
```

像素到局部坐标转换公式：

```python
x_m = (px - left_px) / (right_px - left_px) * width_m
y_m = (bottom_px - py) / (bottom_px - top_px) * height_m
```

注意：这个转换只用于离线整理常量，不作为运行时自动识别流程。

---

## 4. 用户点生成：替换随机用户，保留用户 schema

### 4.1 固定用户点数量

本截图中黄色图钉约 25 个。生成场景时不要再随机生成 50 户。  
`ovi_screenshot_config.yaml` 中设置：

```yaml
users:
  source: ovi_screenshot
  count: 25
```

### 4.2 固定用户点坐标

在 `src/features/ovi_screenshot_features.py` 中定义固定用户点，坐标采用局部米制坐标，原点为红框左下角。

```python
OVI_USER_POINTS_M = [
    (127.7, 899.9),
    (222.6, 826.0),
    (241.3, 781.2),
    (396.9, 686.4),
    (158.3, 658.3),
    (174.9, 643.7),
    (388.6, 640.6),
    (118.9, 636.4),
    (237.1, 559.3),
    (35.9, 518.7),
    (253.7, 506.2),
    (50.4, 500.0),
    (17.2, 468.7),
    (314.9, 394.8),
    (349.2, 357.3),
    (288.0, 334.3),
    (187.3, 327.1),
    (199.8, 266.6),
    (301.5, 238.5),
    (289.4, 213.5),
    (184.2, 209.4),
    (197.7, 123.9),
    (242.3, 116.7),
    (177.0, 35.4),
    (298.3, 21.9),
]
```

### 4.3 用户属性

生成 `users` 图层时必须兼容现有 `FEATURE_LAYER_DEFINITIONS["users"]` schema。建议属性：

```text
user_id: 1..25
user_type: normal_residential 或 high_power_user
settlement_type: ovi_village
cluster_id: null 或 ovi_group_x
load_kw: 默认 7.0，局部高负荷用户 10.0
power_factor: 0.85
phase_type: single
assigned_phase: null
apparent_kva: load_kw / power_factor
importance: 1
elev_m: 从 dtm 按用户点采样
connected_node_id: null
voltage_drop_pct: null
service_group_id: null
service_group_size: 0
is_service_singleton: 0
```

高负荷用户可选取 5 户，建议靠道路或房屋较密集处：

```python
HIGH_POWER_USER_IDS = {4, 9, 11, 15, 19}
```

不要再调用随机用户分布逻辑。  
不要保留“如果识别不到用户就随机生成用户”的 fallback。

---

## 5. DTM 地形生成：核心改动

### 5.1 新增确定性地形生成器

新增文件：

```text
src/terrain/ovi_screenshot_terrain.py
```

提供函数：

```python
def generate_ovi_screenshot_terrain(config: dict) -> np.ndarray:
    ...
```

`src/terrain/terrain_generator.py` 中只增加一个清晰分支：

```python
if terrain_config.get("base_type") == "ovi_screenshot":
    return generate_ovi_screenshot_terrain(config)
```

不要在 `terrain_generator.py` 中堆大量公式和常量。

### 5.2 高程范围

图片中可见等高线主要为：

```text
西侧/左侧：350m ~ 400m
北部局部：300m
中部：300m ~ 350m
东侧/右侧：250m ~ 300m
局部水塘/谷地：约 250m
南部局部：300m
```

配置中采用真实高程量级：

```yaml
terrain:
  base_type: ovi_screenshot
  clip_min: 220
  clip_max: 420
  max_buildable_slope_deg: 24.0
  max_buildable_roughness_m: 10.0
  roughness_window: 5
  smooth_sigma: 4.0
```

### 5.3 地形生成原则

生成 DTM 时不追求测绘级真实，而是追求与图片地貌一致：

```text
1. 整体为山地/丘陵村落台区；
2. 西侧高，东侧低；
3. 左侧靠近红框边缘存在 350~400m 高坡；
4. 中部道路附近形成相对缓坡或沟谷；
5. 东北、东中部水塘附近为低洼平缓区；
6. 用户主要沿道路和缓坡分布；
7. 地形应能给后续线路优化提供坡度、三维长度、净空、成本差异。
```

### 5.4 推荐实现方法

使用确定性控制点插值 + 道路/沟谷修正，不使用随机噪声作为主地形。

#### 5.4.1 控制点

在 `ovi_screenshot_terrain.py` 中定义高程控制点：

```python
ELEVATION_CONTROL_POINTS = [
    # west high ridge
    (0, 0, 350), (0, 200, 380), (0, 450, 400), (0, 700, 360), (0, 909, 310),
    (60, 120, 340), (70, 350, 370), (80, 600, 350), (90, 850, 320),

    # central slope and village belt
    (150, 80, 310), (170, 250, 315), (180, 450, 320), (190, 650, 305), (210, 850, 300),
    (240, 120, 300), (250, 350, 290), (250, 550, 285), (250, 750, 285),

    # east lower area
    (330, 100, 285), (330, 300, 270), (340, 500, 260), (360, 700, 250), (380, 850, 250),
    (428, 0, 300), (428, 250, 260), (428, 550, 250), (428, 909, 270),

    # local pond/valley controls
    (330, 760, 248), (370, 775, 248), (385, 690, 250),
]
```

#### 5.4.2 插值

由于 `requirements.txt` 已有 `scipy`，可以使用：

```python
from scipy.interpolate import RBFInterpolator
```

推荐参数：

```python
rbf = RBFInterpolator(
    xy_points,
    z_values,
    kernel="thin_plate_spline",
    smoothing=1.5,
)
```

插值得到基础高程场。

#### 5.4.3 道路/沟谷修正

道路附近应略微平缓，作为后续线路候选通道的自然趋势。  
定义若干道路中心线，不需要单独作为规划强约束，但可用于地形修正和可视化。

```python
ROAD_CENTERLINES_M = [
    # main north-south road / village belt
    [(125, 900), (150, 760), (170, 630), (210, 520), (245, 380), (285, 220), (300, 20)],

    # west branch
    [(35, 520), (70, 560), (120, 635), (165, 660)],

    # east branch near ponds and houses
    [(250, 560), (310, 610), (380, 650), (410, 690)],

    # south village branch
    [(180, 210), (210, 160), (245, 120), (300, 50)],
]
```

在道路中心线附近进行轻微修正：

```text
道路中心线 0~8m：降低 2~4m，模拟道路削坡或谷线；
道路中心线 8~20m：平滑过渡；
不要把道路挖成深沟。
```

实现上可以计算每个栅格点到道路折线的最短距离，构造：

```python
road_effect = -4.0 * exp(-(distance_to_road / 12.0) ** 2)
dtm += road_effect
```

#### 5.4.4 水塘平整

水塘区域地形应接近水平，定义水塘 polygon 后，对 polygon 内栅格做局部平整：

```text
pond elevation = 248m ~ 252m
```

示例水塘：

```python
WATER_POLYGONS_M = [
    [(318, 790), (365, 790), (370, 750), (325, 742)],   # upper pond
    [(340, 705), (390, 705), (395, 670), (350, 660)],   # middle pond
]
```

处理方式：

```python
dtm[water_mask] = np.minimum(dtm[water_mask], 252.0)
```

#### 5.4.5 平滑与裁剪

最后执行：

```python
from scipy.ndimage import gaussian_filter

dtm = gaussian_filter(dtm, sigma=terrain_config.get("smooth_sigma", 4.0))
dtm = np.clip(dtm, clip_min, clip_max).astype("float32")
```

不要叠加随机 Perlin 噪声。  
如果确实需要细节纹理，只允许使用固定的、幅值很小的确定性正弦纹理：

```python
micro = 1.2 * np.sin(x / 35.0) * np.sin(y / 47.0)
dtm += micro
```

不得使用运行时随机数制造主要地貌。

---

## 6. 障碍与约束图层：替换随机障碍，保留图层名称

虽然本任务核心是地形生成，但当前 `generate-scene` 会同时生成用户与障碍。如果只替换 DTM 而继续随机生成障碍，场景会与奥维截图冲突。因此本次应同步把 `forest`、`water`、`manual_no_build` 改为确定性图层，但不改后续规划算法。

### 6.1 water 图层

对应截图中的水塘，生成 `water` 图层：

```python
WATER_POLYGONS_M = [
    [(318, 790), (365, 790), (370, 750), (325, 742)],
    [(340, 705), (390, 705), (395, 670), (350, 660)],
]
```

字段：

```text
obs_id: 1..n
water_type: pond
forbidden: 1
```

### 6.2 forest 图层

截图左侧和局部山坡为树林/高坡区域。建议生成几个近似 forest polygon：

```python
FOREST_POLYGONS_M = [
    [(0, 0), (75, 0), (85, 230), (55, 520), (30, 760), (0, 909)],       # west forest slope
    [(0, 430), (55, 460), (80, 590), (45, 700), (0, 680)],              # west-mid dense green
    [(245, 520), (315, 560), (325, 650), (260, 690), (225, 610)],       # central-east vegetation belt
]
```

字段：

```text
obs_id: 1..n
density: 0.8
pass_cost: 8.0
forbidden: 1
```

### 6.3 manual_no_build 图层

用于表示截图中明显不适合建设的陡坡、红框边缘外侧或水塘缓冲区。建议生成少量确定性禁建区：

```python
MANUAL_NO_BUILD_POLYGONS_M = [
    [(0, 780), (70, 790), (90, 909), (0, 909)],             # northwest steep edge
    [(370, 650), (428, 655), (428, 760), (390, 735)],       # east pond/steep buffer
]
```

字段：

```text
obs_id: 1..n
source: ovi_screenshot
reason: steep_or_pond_buffer
forbidden: 1
```

### 6.4 roads 可选

当前项目 `features.gpkg` 标准图层中没有 roads。  
如果不想改 schema，不要新增 roads 图层。道路只作为地形生成内部辅助线和场景可视化参考即可。

若确需输出道路，可另存为：

```text
data/vector/ovi_roads.geojson
```

不要把 roads 塞进 `features.gpkg` 的标准 schema，避免影响后续读取逻辑。

---

## 7. 配置文件：新增 ovi_screenshot_config.yaml

新建：

```text
configs/ovi_screenshot_config.yaml
```

内容可以从 `default_config.yaml` 复制，但做以下最小修改。

### 7.1 scene

```yaml
scene:
  width_m: 428.45
  height_m: 909.30
  max_elevation_m: 420
  resolution_m: 1
  origin_x_m: 0
  origin_y_m: 909.30
  crs: EPSG:3857
  seed: 66
```

### 7.2 terrain

```yaml
terrain:
  base_type: ovi_screenshot
  add_perlin_noise: false
  noise_scale: 0.01
  noise_amplitude: 0.0
  noise_octaves: 0
  add_gaussian_hills: false
  hill_count: 0
  hill_sigma_min: 0.15
  hill_sigma_max: 0.30
  valley_ratio: 0.25
  smooth_sigma: 4.0
  clip_min: 220
  clip_max: 420
  max_buildable_slope_deg: 24.0
  max_buildable_roughness_m: 10.0
  roughness_window: 5
```

### 7.3 users

```yaml
users:
  source: ovi_screenshot
  count: 25
  load_groups:
    - name: normal_residential
      count: 20
      load_kw: 7.0
      power_factor: 0.85
      phase_type: single
    - name: high_power_user
      count: 5
      load_kw: 10.0
      power_factor: 0.85
      phase_type: single
  default_phase_type: single
  importance_range: [1, 3]
```

保留旧的随机分布字段也可以，但 ovi 模式下不要读取它们。更干净的做法是：`ovi_screenshot_config.yaml` 只保留实际用到的字段。

### 7.4 obstacles

```yaml
obstacles:
  source: ovi_screenshot
  buffer_from_users_m: 10
```

### 7.5 planning / planning_v2

从默认配置复制。  
原则上不要改算法类参数。  
仅建议根据新台区长宽调整以下配置：

```yaml
planning_v2:
  max_user_path_length_m: 500.0
  enforce_max_user_path_length: true
```

如果希望保留 300m 硬约束，也可以暂不调整，但必须接受优化可能返回不可行。

---

## 8. 主流程改造方式

### 8.1 `generate_scene` 不要重写

`src/main.py` 中的 `generate_scene` 主流程保留：

```text
build_profile
generate_terrain
derive_terrain_layers
generate_users
generate_obstacle_layers
rasterize_forbidden_mask
derive_terrain_layers
build_cost_surface
write rasters
write features.gpkg
plot scene
plot terrain 3d
```

只让各 generator 根据 config source/base_type 选择确定性奥维实现。

### 8.2 `terrain_generator.py`

保持入口简洁：

```python
def generate_terrain(config: dict[str, Any]) -> np.ndarray:
    terrain_cfg = config["terrain"]
    base_type = terrain_cfg.get("base_type")

    if base_type == "ovi_screenshot":
        from src.terrain.ovi_screenshot_terrain import generate_ovi_screenshot_terrain
        return generate_ovi_screenshot_terrain(config)

    # 保留既有随机地形逻辑，用于 default_config 和旧测试
    ...
```

不要为 `ovi_screenshot` 写 fallback，例如：

```python
# 禁止
try:
    return generate_ovi_screenshot_terrain(config)
except Exception:
    return generate_random_terrain(config)
```

如果配置错误，应直接抛异常。

### 8.3 `users_generator.py`

入口示例：

```python
def generate_users(config, *, dtm, valid_mask, transform, crs):
    if config.get("users", {}).get("source") == "ovi_screenshot":
        from src.features.ovi_screenshot_features import generate_ovi_users
        return generate_ovi_users(config, dtm=dtm, transform=transform, crs=crs)

    # 保留既有随机用户逻辑
    ...
```

### 8.4 `obstacles_generator.py`

入口示例：

```python
def generate_obstacle_layers(config, *, scene_bounds, crs, users):
    if config.get("obstacles", {}).get("source") == "ovi_screenshot":
        from src.features.ovi_screenshot_features import generate_ovi_obstacles
        return generate_ovi_obstacles(config, crs=crs)

    # 保留既有随机障碍逻辑
    ...
```

---

## 9. 代码清理要求

需要清理：

```text
1. 旧版针对截图的临时代码；
2. 运行时自动识别红框、识别图钉的实验代码；
3. 大量 fallback 到随机场景的兜底逻辑；
4. 与当前 features.gpkg schema 不兼容的临时 geojson 输出；
5. 重复的用户点常量；
6. 与本次任务无关的规划算法重构。
```

不要清理：

```text
1. default_config.yaml；
2. 旧随机场景生成能力；
3. 现有测试依赖的默认随机场景；
4. V2 优化器；
5. terrain derivatives；
6. raster/vector IO。
```

理由：默认随机场景仍然是测试和回归验证基线，不能为了奥维截图场景把它删除。

---

## 10. 测试要求

新增：

```text
tests/test_ovi_screenshot_scene.py
```

至少测试以下内容。

### 10.1 DTM 尺寸

```python
def test_ovi_terrain_shape():
    config = load_config(Path("configs/ovi_screenshot_config.yaml"))
    dtm = generate_terrain(config)
    assert dtm.shape == (910, 428) or dtm.shape == (909, 428)
```

具体 shape 以 `build_profile` 计算结果为准。不要在测试里模糊匹配多个尺寸，Codex 实现后固定一个确定 shape。

### 10.2 高程范围

```python
assert dtm.min() >= 220
assert dtm.max() <= 420
assert dtm[:, :60].mean() > dtm[:, -80:].mean()
```

即左侧整体高于右侧。

### 10.3 用户点数量

```python
users = generate_ovi_users(...)
assert len(users) == 25
assert set(users.columns) 包含 FEATURE_LAYER_DEFINITIONS["users"] 需要的字段
```

### 10.4 障碍图层

```python
obstacles = generate_ovi_obstacles(...)
assert len(obstacles["water"]) >= 2
assert len(obstacles["forest"]) >= 1
assert len(obstacles["manual_no_build"]) >= 1
```

### 10.5 完整流程冒烟测试

可以加入一个轻量命令测试，至少保证：

```bash
python -m src.main generate-scene --config configs/ovi_screenshot_config.yaml
python -m src.main derive-terrain --config configs/ovi_screenshot_config.yaml
```

优化器完整测试耗时较长，可以不放入默认单元测试，但本地必须手工运行一次：

```bash
python -m src.main optimize-plan --config configs/ovi_screenshot_config.yaml
python -m src.main plot-plan --config configs/ovi_screenshot_config.yaml
```

---

## 11. 验收标准

完成后应产生以下文件：

```text
data/terrain/dtm.tif
data/terrain/slope.tif
data/terrain/aspect.tif
data/terrain/roughness.tif
data/terrain/cost_base.tif

data/masks/forbidden_mask.tif
data/masks/buildable_mask.tif

data/vector/features.gpkg

data/outputs/plots/scene_2d.png
data/outputs/plots/terrain_3d_preview.png
data/outputs/plots/terrain_3d_preview.html

data/outputs/plans/optimization_summary.json
data/outputs/plots/optimized_plan_2d.png
data/outputs/plots/optimized_plan_3d_static.png
data/outputs/plots/optimized_plan_3d_dynamic.html
```

其中：

```text
features.gpkg/users: 25 户用户
features.gpkg/water: 固定水塘
features.gpkg/forest: 固定林地/高坡
features.gpkg/manual_no_build: 固定禁建区
dtm.tif: 与截图等高线趋势一致的丘陵台区地形
```

可视化效果应满足：

```text
1. 台区范围为狭长矩形，长边约 909m；
2. 用户沿道路/村落带分布，而非随机撒点；
3. 西侧和左下侧明显较高；
4. 东侧和水塘区域较低；
5. 优化结果仍沿用现有 V2 规划逻辑生成配变、线路和杆塔；
6. 不出现与截图完全无关的随机树林、水塘、禁建区。
```

---

## 12. 给 Codex 的执行指令

可以直接把下面这段给 Codex：

```text
请按 docs/Codex执行方案_基于奥维截图重建台区仿真地形_v0.2.md 执行。

本次任务只针对 generate-scene 阶段做改造：把默认随机地形/随机用户/随机障碍替换为 ovi_screenshot 模式下的确定性奥维截图台区场景。后续 derive-terrain、optimize-plan、plot-plan 的主流程和 V2 优化器不要重写。

请新增 configs/ovi_screenshot_config.yaml、src/terrain/ovi_screenshot_terrain.py、src/features/ovi_screenshot_features.py，并在 terrain_generator.py、users_generator.py、obstacles_generator.py 中增加清晰分支。不要做运行时通用图片识别，不要写 try/except fallback 到随机场景。如果 ovi 配置错误，应直接报错。

保持 features.gpkg 标准图层和字段 schema 不变。生成 428.45m × 909.30m、1m 分辨率的仿真 DTM，用户点固定为方案中的 25 个坐标，水塘/林地/禁建区使用确定性 polygon。后续优化规划逻辑基本保留，只允许在 ovi_screenshot_config.yaml 中按新场景尺寸做配置级参数调整。
```
