# 台区配电线路优化项目开发计划（供 Codex 执行）

## 1. 项目目标

构建一个 Python 工程，用于生成并管理一个台区仿真场景，作为后续配电变压器选址、杆塔布置、线路寻优的统一数据底座。

本项目当前阶段只完成以下内容：

1. 生成 **5km × 5km × 500m** 的随机地形
2. 地形主格式采用 **DTM GeoTIFF**
3. 分辨率固定为 **1m**
4. 随机生成 **50 个用户点**
5. 随机生成障碍物（树林、水区、人工禁建区等）
6. 支持人工补充和修改障碍区
7. 生成标准化输出，供后续优化算法直接读取
8. 生成基础可视化结果，便于人工检查

---

## 2. 核心设计原则

### 2.1 地形主格式

地形主数据采用：

- `DTM GeoTIFF`

其角色定义为：

- 作为整个项目的**唯一主地形底图**
- 用于后续高程查询、坡度计算、通行代价构建、杆塔可建设性分析、线路寻径

### 2.2 非地形对象单独存储

以下内容**不直接写入 DTM 栅格本体**，而是独立存储：

- 用户点
- 树林
- 水区
- 人工禁建区
- 后续候选配变点
- 后续候选杆塔点
- 后续规划线路

这些内容统一存入：

- `GeoPackage (.gpkg)`

### 2.3 派生栅格单独输出

从 DTM 派生的各类分析结果单独输出为 GeoTIFF，例如：

- 坡度栅格
- 坡向栅格
- 粗糙度栅格
- 禁行区掩膜
- 综合代价栅格

---

## 3. 本阶段明确结论

### 结论

本项目后续所有代码实现，都应基于以下数据架构：

- **地形主底图：DTM GeoTIFF**
- **对象与障碍库：GeoPackage**
- **约束与代价栅格：GeoTIFF**

不得将所有信息混在一个文件里。

---

## 4. 场景基础参数

## 4.1 空间范围

- 宽度：`5000 m`
- 高度：`5000 m`
- 最大相对高差控制目标：`500 m`

## 4.2 分辨率

- 分辨率：`1 m`

## 4.3 栅格尺寸

由于范围为 `5000m × 5000m`，分辨率为 `1m`，因此 DTM 栅格尺寸固定为：

- `5000 × 5000`

## 4.4 高程范围

推荐将高程裁剪到：

- 最小值：`0 m`
- 最大值：`500 m`

---

## 5. 输出文件结构

```text
project_root/
├─ data/
│  ├─ terrain/
│  │  ├─ dtm.tif
│  │  ├─ slope.tif
│  │  ├─ aspect.tif
│  │  ├─ roughness.tif
│  │  └─ cost_base.tif
│  ├─ masks/
│  │  ├─ forbidden_mask.tif
│  │  └─ buildable_mask.tif
│  ├─ vector/
│  │  └─ features.gpkg
│  └─ outputs/
│     ├─ plots/
│     └─ plans/
├─ configs/
│  └─ default_config.yaml
├─ src/
│  ├─ io/
│  │  ├─ raster_io.py
│  │  └─ vector_io.py
│  ├─ terrain/
│  │  ├─ terrain_generator.py
│  │  ├─ terrain_derivatives.py
│  │  └─ terrain_validator.py
│  ├─ features/
│  │  ├─ users_generator.py
│  │  ├─ obstacles_generator.py
│  │  └─ manual_constraints.py
│  ├─ planning/
│  │  ├─ candidate_generator.py
│  │  ├─ cost_surface.py
│  │  └─ optimizer_stub.py
│  ├─ viz/
│  │  └─ plot_scene.py
│  └─ main.py
├─ tests/
│  ├─ test_terrain.py
│  ├─ test_features.py
│  └─ test_io.py
├─ requirements.txt
└─ README.md
```

---

## 6. DTM GeoTIFF 数据规范

## 6.1 文件位置

- `data/terrain/dtm.tif`

## 6.2 基本要求

- 单波段
- 数据类型：`float32`
- 单位：米
- `nodata = -9999`
- 分辨率：`1 m`
- 宽高：`5000 × 5000`

## 6.3 读写要求

代码中必须显式管理以下参数：

- `driver`
- `width`
- `height`
- `count`
- `dtype`
- `crs`
- `transform`
- `nodata`

## 6.4 写出建议

GeoTIFF 写出建议：

- 启用压缩
- 启用分块
- 预留金字塔概览接口
- 所有派生栅格必须与 `dtm.tif` 保持完全一致的：
  - 尺寸
  - 分辨率
  - 仿射变换
  - 坐标系

---

## 7. GeoPackage 数据规范

## 7.1 文件位置

- `data/vector/features.gpkg`

## 7.2 必须包含的图层

### `users`
几何类型：`Point`

字段：

- `user_id`：`int`
- `load_kw`：`float`
- `phase_type`：`str`
- `importance`：`int`
- `elev_m`：`float`

### `forest`
几何类型：`Polygon`

字段：

- `obs_id`：`int`
- `density`：`float`
- `pass_cost`：`float`
- `forbidden`：`int`

### `water`
几何类型：`Polygon`

字段：

- `obs_id`：`int`
- `water_type`：`str`
- `forbidden`：`int`

### `manual_no_build`
几何类型：`Polygon`

字段：

- `obs_id`：`int`
- `source`：`str`
- `reason`：`str`
- `forbidden`：`int`

### 预留图层

以下图层本阶段可以先建空图层：

#### `candidate_transformer`
几何类型：`Point`

#### `candidate_poles`
几何类型：`Point`

#### `planned_lines`
几何类型：`LineString`

---

## 8. 配置文件要求

必须采用配置文件驱动，不允许把场景参数硬编码进脚本。

建议配置文件：

- `configs/default_config.yaml`

建议模板如下：

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
  noise_amplitude: 25
  add_gaussian_hills: true
  hill_count: 12
  smooth_sigma: 3.0
  clip_min: 0
  clip_max: 500

users:
  count: 50
  min_spacing_m: 100
  distribution_mode: clustered
  load_kw_range: [2, 25]
  importance_range: [1, 3]

obstacles:
  forest_count: 8
  water_count: 3
  manual_no_build_count: 2
  min_area_m2: 2000
  max_area_m2: 80000
  buffer_from_users_m: 20

planning:
  transformer_candidate_step_m: 100
  pole_candidate_step_m: 50
```

---

## 9. 模块开发顺序

## Phase 1：基础 I/O 能力

### 目标
先把 GeoTIFF 和 GeoPackage 的稳定读写能力建立起来。

### 任务

#### `src/io/raster_io.py`
实现：

- 写 GeoTIFF
- 读 GeoTIFF
- 读取 metadata
- 保证 transform、crs、nodata 正确传递

#### `src/io/vector_io.py`
实现：

- 写 GeoPackage 图层
- 读取图层
- 追加图层
- 覆盖图层
- 创建空图层

#### `src/main.py`
实现 CLI 命令入口：

- `generate-scene`
- `derive-terrain`
- `plot-scene`

### 验收

- 能创建 `dtm.tif`
- 能创建 `features.gpkg`
- 所有文件可以再次被 Python 读入

---

## Phase 2：随机地形生成

### 目标
生成连续、自然、可复现的 1m 分辨率地形。

### 任务

#### 2.1 基础地形
生成一个基础马鞍面，作为整体趋势地形。

#### 2.2 地形扰动
在基础面上叠加：

- Perlin 噪声或 fBm 噪声
- 若干局部高斯丘陵
- 若干局部低洼

#### 2.3 平滑
使用高斯滤波对地形进行平滑，避免明显锯齿或非自然突变。

#### 2.4 裁剪
将地形高程裁剪到 `0~500m`。

#### 2.5 输出
写出：

- `data/terrain/dtm.tif`

### 要求

- 支持固定 `seed`
- 同一 `seed` 下结果必须一致
- 地形应具备一定起伏，但不能过于极端
- 地形应适合后续布线和障碍叠加分析

### 验收

- 输出 `dtm.tif`
- 输出预览图 `terrain_preview.png`
- 输出基础统计信息：
  - 最小高程
  - 最大高程
  - 平均高程
  - 高程标准差

---

## Phase 3：地形派生栅格

### 目标
从 DTM 派生后续优化需要的基础分析栅格。

### 任务

生成：

- `slope.tif`
- `aspect.tif`
- `roughness.tif`
- `buildable_mask.tif`

### 规则建议

#### 坡度
- 计算每个像元坡度
- 作为线路施工和杆塔布设的重要成本因子

#### 坡向
- 先保留结果，不一定立即参与优化
- 方便后续扩展地形朝向分析

#### 粗糙度
- 反映局部起伏程度
- 作为施工难度附加因子

#### 可建设掩膜
根据坡度阈值、粗糙度阈值初步筛选可建设区。

### 验收

- 所有派生栅格与 `dtm.tif` 尺寸、分辨率、坐标完全一致
- 所有文件成功写出并可读取

---

## Phase 4：用户点生成

### 目标
生成 50 个用户点，并满足空间约束。

### 任务

#### 4.1 生成规则
- 总数固定为 50
- 支持两种模式：
  - `uniform`
  - `clustered`

#### 4.2 约束
- 用户点之间应满足最小间距
- 不得落在禁行区、水区、明显不可建设区
- 若落在无效区，应自动重采样

#### 4.3 属性赋值
为每个用户生成：

- `user_id`
- `load_kw`
- `phase_type`
- `importance`
- `elev_m`

其中 `elev_m` 必须从 `dtm.tif` 采样获得。

### 输出
写入 `features.gpkg` 的 `users` 图层。

### 验收

- 用户数严格为 50
- 所有用户点都在有效区域
- 所有用户点都能成功采样高程

---

## Phase 5：障碍物生成

### 目标
构建影响线路规划的障碍体系。

### 障碍类型

#### 1. 树林
- 多边形
- 可设为高代价区或禁行区

#### 2. 水区
- 多边形
- 默认禁行区

#### 3. 人工禁建区
- 多边形
- 可随机生成，也可后续人工导入覆盖

### 任务

#### 5.1 自动生成
用以下方法生成随机障碍形状：

- 椭圆
- 随机扰动多边形
- buffer / union / difference

#### 5.2 合法性约束
- 必须限制在场景边界内
- 不得生成无效几何
- 尽量避免完全覆盖用户密集区
- 如与用户冲突，可重采样或裁切

#### 5.3 写出
写入以下图层：

- `forest`
- `water`
- `manual_no_build`

#### 5.4 栅格化
根据禁行区生成：

- `data/masks/forbidden_mask.tif`

### 验收

- 三类障碍均成功生成
- 禁行区掩膜与矢量障碍逻辑一致
- 用户点不得落入禁行区

---

## Phase 6：人工约束导入

### 目标
支持人工补充或覆盖障碍区。

### 要求

实现以下任一或全部方式：

#### 方式一
读取外部文件：

- `manual_constraints.geojson`

#### 方式二
直接更新：

- `features.gpkg` 中的 `manual_no_build` 图层

### 行为规则

- 外部导入后，应自动融合进禁行区体系
- 更新后应重新生成 `forbidden_mask.tif`

### 验收

- 人工导入成功后，禁行区栅格会同步更新

---

## Phase 7：综合可视化

### 目标
生成便于人工检查的场景图件。

### 至少输出以下图件

1. `terrain_preview.png`
2. `slope_preview.png`
3. `scene_overview.png`
4. `terrain_with_features.png`
5. `forbidden_mask.png`

### 图件要求

- 有标题
- 有图例
- 有比例信息
- 用户、树林、水区、禁建区叠加关系清晰
- 图片适合直接人工检查

---

## Phase 8：为后续优化算法预留接口

### 目标
本阶段不实现完整优化算法，但要把接口结构定下来。

### 模块

#### `src/planning/cost_surface.py`
负责构建综合代价栅格，输入包括：

- 坡度
- 粗糙度
- 水区禁行
- 树林通行代价
- 人工禁建区

输出：

- `cost_base.tif`

#### `src/planning/candidate_generator.py`
负责生成：

- 候选配变点
- 候选杆塔点

#### `src/planning/optimizer_stub.py`
只定义输入输出接口，不强行实现最终算法。

### 未来算法接口输入

- `dtm.tif`
- `slope.tif`
- `roughness.tif`
- `forbidden_mask.tif`
- `features.gpkg`

### 未来算法接口输出

- `candidate_transformer`
- `candidate_poles`
- `planned_lines`

---

## 10. 推荐依赖

`requirements.txt` 至少包含：

```text
numpy
scipy
rasterio
geopandas
shapely
pyproj
matplotlib
pyyaml
networkx
```

可选增强依赖：

```text
scikit-image
loguru
rich
```

---

## 11. 编码规范

Codex 生成代码时必须遵守以下规则：

1. 所有核心函数必须写类型注解
2. 所有核心函数必须写 docstring
3. 所有随机过程必须支持 `seed`
4. 所有路径必须通过配置文件或参数传入
5. 不允许写死本地绝对路径
6. 所有阶段必须可以单独运行
7. 对异常情况必须显式报错
8. 栅格和矢量的空间范围不一致时必须检测并报错
9. 所有输出文件名必须稳定、规范、可预测
10. 项目必须带有一个可直接运行的 demo

---

## 12. 测试要求

至少编写以下测试：

### `tests/test_io.py`
测试：

- GeoTIFF 读写
- GeoPackage 图层读写
- metadata 保持正确

### `tests/test_terrain.py`
测试：

- 地形生成尺寸正确
- 高程范围正确
- 同一 seed 可复现

### `tests/test_features.py`
测试：

- 用户数量正确
- 用户不落入禁区
- 障碍图层可成功生成
- 掩膜与障碍逻辑一致

---

## 13. 验收标准

当首版代码完成后，必须满足以下条件：

### A. 数据成果完整
必须成功生成：

- `data/terrain/dtm.tif`
- `data/terrain/slope.tif`
- `data/terrain/aspect.tif`
- `data/terrain/roughness.tif`
- `data/masks/forbidden_mask.tif`
- `data/vector/features.gpkg`

### B. 场景正确
必须满足：

- 地形范围为 `5km × 5km`
- 分辨率为 `1m`
- 栅格尺寸为 `5000 × 5000`
- 高程范围在 `0~500m`
- 用户数为 `50`
- 至少包含树林、水区、人工禁建区三类障碍

### C. 稳定性
必须满足：

- 同一 seed 结果可复现
- 不同 seed 下场景明显不同
- 所有输出均可再次读入

### D. 可视化
必须输出至少 3 张可检查图件，且叠加关系正确。

---

## 14. 给 Codex 的直接执行指令

将以下文字作为 Codex 的直接任务描述：

```text
请为一个“台区配电线路优化”的 Python 工程生成首版代码框架。

项目要求如下：

1. 地形主底图采用 DTM GeoTIFF
2. 场景范围为 5km × 5km
3. 分辨率固定为 1m，因此 DTM 栅格尺寸固定为 5000 × 5000
4. 高程范围控制在 0~500m
5. 随机生成 50 个用户点
6. 随机生成树林、水区、人工禁建区等障碍物
7. 用户、障碍、后续规划结果统一写入 GeoPackage
8. 从 DTM 派生坡度、坡向、粗糙度、禁行区掩膜、基础代价栅格
9. 生成基础可视化图件
10. 为后续配变选址、杆塔布置、线路优化预留接口

技术要求：
- 使用 Python
- 使用 numpy、scipy、rasterio、geopandas、shapely、matplotlib、pyyaml、networkx
- 使用配置文件驱动参数
- 所有随机过程支持 seed
- 所有模块带类型注解和 docstring
- 生成完整工程骨架、requirements.txt、README.md、main.py、测试文件和 demo 配置
- 先保证工程结构清晰、数据规范统一、结果可复现
- 不要只写一个脚本，要写成可扩展项目结构

输出目录和文件命名必须严格遵守以下规范：
- data/terrain/dtm.tif
- data/terrain/slope.tif
- data/terrain/aspect.tif
- data/terrain/roughness.tif
- data/masks/forbidden_mask.tif
- data/vector/features.gpkg

features.gpkg 中至少包含以下图层：
- users
- forest
- water
- manual_no_build
- candidate_transformer
- candidate_poles
- planned_lines
```

---

## 15. 最终决策

本项目当前阶段正式采用以下数据架构：

- **DTM GeoTIFF：主地形底图**
- **GeoPackage：用户、障碍、方案成果库**
- **GeoTIFF：派生约束与代价栅格**

后续所有代码和算法设计，均以该架构为准，不再改动主格式。
