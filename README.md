# DistributionNetOptimizer

用于配电网规划前期场景构建与三维布线优化的 Python 项目。当前版本已经支持：

- 随机生成三维地形 `DTM`
- 生成坡度、坡向、粗糙度、可建设区、基础成本栅格
- 随机生成用户、树林、水区、人工禁区
- 将结果统一写入 `GeoPackage`
- 基于三维地形执行配变选址、380V 低压径向网络规划
- 进行 ABC 相别分配、压降校核、线路净空校核
- 输出 2D 静态图、3D 静态图、3D 动态交互图

## 1. 当前实现重点

### 1.1 场景生成

项目可以从配置文件出发，一键生成：

- 地形栅格
- 用户点
- 树林、水区、人工禁区
- 禁区掩膜和可建设区
- 二维与三维预览图

### 1.2 配电网优化

当前优化器已经实现：

- 单台配变选址
- 低压主干 ABCN 共线路径规划
- 连续可行域驱动的稀疏可见图路径搜索
- 路径生成后的自动后插杆
- 用户从杆塔分接接入
- 单相用户 ABC 相别分配
- 三相负荷基本平衡优化
- 压降校核（基于负荷电流和线路阻抗精确计算）
- 线路全线最小离地净空校核
- 净空不足时自动增设杆塔修复

### 1.3 低压共享主干规则

当前版本的重要建模规则是：

- 低压线路不是“每户一条独立支路”
- 优化器会优先生成共享的低压主干树
- 每条低压主干线路按 `A/B/C/N` 同路径敷设
- 用户只能作为终端
- 用户通过某根已有杆塔分接接入
- 允许多个用户从同一根杆塔或相邻杆塔引下接户线
- 优化器会优先复用已有杆塔和已有线段
- 不允许出现平行重复的主干线路

也就是说，当前模型强调的是：

1. 先形成共享低压骨干。
2. 再从骨干杆塔向用户分接。
3. 不把每个用户都当成独立主干终点去拉一整条支路。

## 2. 安装

```bash
python -m pip install -r requirements.txt
```

依赖文件见 [requirements.txt](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/requirements.txt)。

## 3. 目录结构

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

## 4. 常用命令

### 4.1 生成基础场景

```bash
python -m src.main generate-scene --config configs/default_config.yaml
```

小场景调试建议：

```bash
python -m src.main generate-scene --config configs/demo_small.yaml
```

### 4.2 重建地形派生结果

```bash
python -m src.main derive-terrain --config configs/default_config.yaml
```

### 4.3 重绘 2D 场景图

```bash
python -m src.main plot-scene --config configs/default_config.yaml
```

### 4.4 生成 3D 地形预览

```bash
python -m src.main plot-terrain-3d --config configs/default_config.yaml
```

### 4.5 刷新人工禁区

```bash
python -m src.main refresh-manual --config configs/default_config.yaml --manual-geojson path/to/manual_constraints.geojson
```

### 4.6 执行优化

```bash
python -m src.main optimize-plan --config configs/default_config.yaml
```

输出包括：

- `candidate_transformer`
- `candidate_poles`
- `planned_lines`
- `optimization_summary.json`
- `optimized_plan_2d.png`
- `optimized_plan_3d_static.png`
- `optimized_plan_3d_dynamic.html`

### 4.7 只重绘优化结果

```bash
python -m src.main plot-plan --config configs/default_config.yaml
```

## 5. 默认配置

默认配置文件是 [configs/default_config.yaml](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/configs/default_config.yaml:1)。

### 5.1 场景范围

默认场景已经调整为：

- `1000m x 1000m`
- `origin_x_m = 0`
- `origin_y_m = 1000`

### 5.2 默认用户负荷

默认用户组为：

- `40` 户 `7kW`
- `10` 户 `12kW`
- 功率因数统一 `0.85`
- 默认均为单相用户

### 5.3 默认杆高与净空

为了避免杆塔过密，默认净空和高度做了适度放宽：

- `lv_pole_height_m = 9.5`
- `transformer_lv_connection_height_m = 8.5`
- `lv_ground_clearance_m = 4.0`
- `service_ground_clearance_m = 2.3`

### 5.4 默认线路参数

压降计算和优化使用的默认参数：

- `line_resistance_ohm_per_km = 0.6`（典型低压线路电阻）
- `line_reactance_ohm_per_km = 0.35`（典型低压线路电抗）
- `low_voltage_phase_v = 230.0`（相电压）
- `voltage_drop_max_pct = 7.0`（允许最大压降）
- `voltage_drop_optimization_weight = 3.0`（压降优化权重）

## 6. 优化器规则

优化器实现位于 [src/planning/optimizer.py](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/src/planning/optimizer.py:1)。

### 6.1 低压网络分层

当前已区分：

- 低压主干：`lv_line`
- 接户线：`service_drop`
- 低压杆：`lv_pole`
- 净空修复杆：`clearance_repair`

### 6.2 用户接入方式

当前版本采用“共享主干 + 杆塔分接”的接入策略：

- 用户不进入主干路径搜索
- 用户不是线路连接点
- 用户只挂接到某根可行杆塔
- 主干先规划，再由用户分接
- 接户线只承担最后一段接入

### 6.3 路径搜索与后插杆

当前版本已经不再使用“规则栅格正交最短路”作为主路由器。

当前实现方式是：

- 先在连续可行域外包出可通行区域
- 在连续可行域中提取稀疏锚点
- 用可见图方式连接线视距可达的锚点
- 在该几何图上完成低压共享主干搜索
- 再沿结果折线按档距自动后插杆
- 同时优先复用已有共享主干和已有杆塔

这意味着它比旧版规则格网更接近连续空间路径，也更容易绕障碍形成自然折线，但它仍然不是数学意义上的“连续空间全局最优”。

这意味着：

- 相比旧版规则格网，线路走向会明显更自然
- 仍会受到 `path_search_step_m`、锚点密度、共享主干复用策略和障碍采样精度影响
- 当前是“连续可行域近似优化 + 稀疏几何图搜索”，不是严格连续变量全局最优求解
- 如果后续继续升级，可以在此基础上增加路径平滑、Theta* 微调或局部连续位置优化

### 6.4 三相分配

单相用户由优化器自动分配到 `A/B/C` 相：

- `A 相 = 黄`
- `B 相 = 绿`
- `C 相 = 红`

优化目标同时考虑：

- 配变总相平衡
- 每一段共享低压主干线路的三相平衡

说明：

- 这里的“线路三相平衡”指的是共享的 `lv_line` 主干段
- `service_drop` 是单相或单户引下线，不作为 `ABC` 平衡目标段

### 6.5 净空修复

每条线路段都会沿线采样地形，检查导线与地面的最小净空。

若净空不足，优化器会尝试：

1. 在问题位置附近寻找可建设点。
2. 自动增设杆塔。
3. 将问题线段拆分为多段后重新校核。

当前版本会在低压共享主干折线上执行"后插杆"：

1. 路径先按几何折线求出。
2. 若某段平面跨度超过档距上限，则沿该段自动增设中间杆塔。
3. 再对拆分后的线段继续做净空和电气校核。

如果仍无法满足，则该段会被标记为违规，并写入汇总结果。

### 6.6 压降计算

压降计算采用基于负荷电流和线路阻抗的精确方法：

**计算公式：**

```
ΔU% = (I × (R × cosφ + X × sinφ) × L × 100) / V
```

其中：
- `I` = 电流（A）= S(kVA) × 1000 / V_phase
- `R` = 线路电阻（Ω/km），默认 0.6 Ω/km
- `X` = 线路电抗（Ω/km），默认 0.35 Ω/km
- `L` = 线路长度（km）
- `V` = 相电压（230V）
- `cosφ` = 功率因数（默认 0.85）

**计算逻辑：**

1. 从变压器出发，沿树形网络向下遍历
2. 每段线路的压降由该段承载的下游负荷决定
3. 用户总压降 = 从变压器到用户的所有线路段压降之和
4. 上游用户的负荷会影响下游所有用户的压降

**配置参数：**

- `line_resistance_ohm_per_km`：线路电阻（Ω/km）
- `line_reactance_ohm_per_km`：线路电抗（Ω/km）
- `low_voltage_phase_v`：相电压（V）
- `voltage_drop_max_pct`：允许最大压降百分比
- `voltage_drop_optimization_weight`：压降优化权重（默认 0.5）

### 6.7 压降优化目标

除了将压降作为约束条件外，优化器还将**平均压降最小化**纳入优化目标：

**优化目标函数：**

```
Score = 成本 + w × 压降惩罚
```

**压降惩罚公式：**

```
if distance > 120m:
    vdrop_penalty = ((distance - 120) / 100)² × √(S/10) × w × 200
else:
    vdrop_penalty = 0
```

其中：
- **distance** = 配变到接入点的距离（m）
- **S** = 用户视在功率（kVA）
- **w** = `voltage_drop_optimization_weight`（默认 3.0）
- **阈值** = 120m（低于此值不惩罚）

**优化策略：**

1. **距离压降惩罚**：配变到接入点 > 120m 时，按平方增长施加惩罚
2. **角度差异惩罚**（关键改进）：当用户"接入已有主干"时，计算用户方向与接入点方向的夹角
   - 夹角 ≤ 45°：无额外惩罚（顺路共享）
   - 夹角 45°-90°：中等惩罚（需要绕路）
   - 夹角 > 90°：强惩罚（严重绕路，应建立新分支）
3. **效果**：自然形成辐射状网络，每个方向的用户倾向于独立分支

**效果：**

- 避免远端用户通过超长路径串联接入
- 鼓励为远离主负荷中心（> 400m）的用户群建立独立分支
- 保持同区域内用户的共享接入行为
- 权重越大，越倾向于生成多条独立短路径

## 7. GeoPackage 图层

统一 schema 定义见 [src/io/vector_io.py](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/src/io/vector_io.py:8)。

主要图层包括：

- `users`
- `forest`
- `water`
- `manual_no_build`
- `candidate_transformer`
- `candidate_poles`
- `planned_lines`

### 7.1 `users` 关键字段

- `load_kw`
- `power_factor`
- `phase_type`
- `assigned_phase`
- `apparent_kva`
- `elev_m`
- `connected_node_id`
- `voltage_drop_pct`

### 7.2 `planned_lines` 关键字段

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

- `A`：黄
- `B`：绿
- `C`：红

### 8.2 线路颜色

- `lv_line`：深灰色
- `service_drop`：橙色

### 8.3 杆塔颜色

- `lv_pole`：青色
- `clearance_repair`：粉色

### 8.4 3D 图形显示

当前 3D 图中不仅显示杆塔顶点，还会绘制杆塔完整立杆，从地面一直画到杆顶，便于检查：

- 杆塔是否落在障碍区外
- 杆高是否合理
- 线路是否从杆顶出线
- 杆塔与地形关系是否正常

输出文件：

- `data/outputs/plots/optimized_plan_2d.png`
- `data/outputs/plots/optimized_plan_3d_static.png`
- `data/outputs/plots/optimized_plan_3d_dynamic.html`

## 9. 测试

```bash
python -m pytest -q
```

当前测试覆盖重点包括：

- 用户 `load_groups` 接口
- 优化器基础输出
- 用户相别分配
- 接户线长度约束
- 用户不作为主干连接点
- 多户共享低压主干与共享接入杆
- 连续域可见图中的对角/视距连接
- 三维长度校验
- 3D 图输出

## 10. 注意事项

### 10.1 Windows 下 GeoPackage 文件锁

如果 `data/vector/features.gpkg` 正在被 QGIS、ArcGIS、Notebook 或其他程序占用，以下命令可能失败：

- `generate-scene`
- `optimize-plan`
- 任何需要重写 `features.gpkg` 的命令

出现“另一个程序正在使用此文件”时，请先关闭占用程序后重试。

### 10.2 当前仍是规划级模型

当前版本适合做方案生成与规划筛选，不直接替代正式施工设计。尚未纳入：

- 精细弧垂力学模型
- 转角杆 / 耐张杆结构设计
- 导线型号与机械强度校核

## 11. 相关规划文档

- [PLAN_geo.md](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/PLAN_geo.md)
- [PLAN_opt.md](C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/PLAN_opt.md)
