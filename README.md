# DistributionNetOptimizer

用于配电网规划前期场景构建与 V2 低压配电优化的 Python 项目。

当前版本已经按 `PLAN_opt_V2.md` 完成优化器主架构重构，旧的“连续域路径搜索 + 可见图 + 路径后插杆”不再参与执行链路。项目现在统一采用：

`候选走廊图 -> 径向树选边 -> 三相分配 -> 潮流/线损评估 -> 杆塔派生`

## 核心能力

- 生成地形、坡度、粗糙度、可建设区与禁建区
- 生成用户、树林、水域、人工禁建区等场景要素
- 基于候选走廊图构建工程可行的低压布线候选网络
- 在走廊图上选择配变位置、共享主干与分支树结构
- 对单相用户进行 `A/B/C` 相别优化
- 进行径向潮流近似评估，输出压降、线损与相不平衡指标
- 基于已选边自动派生杆塔与最终 `planned_lines` 图层
- 在终端显示优化进度条，观察候选评估和局部搜索进度
- 输出 `GeoPackage`、2D 图和 3D 图

## V2 优化架构

`src/planning/` 的主要模块如下：

```text
src/planning/
  optimizer_v2.py            # V2 总调度入口
  corridor_graph.py          # 候选走廊图构建
  transformer_candidates.py  # 配变候选预筛
  attachment_model.py        # 用户接入候选
  radial_tree_milp.py        # 径向树选边近似求解
  phase_assignment.py        # 三相分配优化
  bfs_power_flow.py          # 径向潮流与压降评估
  loss_eval.py               # 线损成本折算
  voltage_eval.py            # 约束检查与不平衡惩罚
  pole_generation.py         # 杆塔与线路图层派生
  geometry_constraints.py    # 走廊与净空几何约束
  neighborhood_search.py     # 局部重挂接改进
  progress.py                # 终端进度条显示
  summary_v2.py              # 输出汇总
```

说明：

- [src/main.py](/C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/src/main.py:1) 的 `optimize-plan` 已直接调用 `optimize_distribution_network_v2(...)`
- [src/planning/optimizer.py](/C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/src/planning/optimizer.py:1) 仅保留项目级统一入口名，不再包含旧算法逻辑
- `planning_v2` 是新的优化配置块；`planning` 里的成本、电气和几何常量仍会被复用

## 安装

```bash
python -m pip install -r requirements.txt
```

## 常用命令

生成完整场景：

```bash
python -m src.main generate-scene --config configs/default_config.yaml
```

重建地形派生结果：

```bash
python -m src.main derive-terrain --config configs/default_config.yaml
```

执行 V2 优化：

```bash
python -m src.main optimize-plan --config configs/default_config.yaml
```

重绘优化结果：

```bash
python -m src.main plot-plan --config configs/default_config.yaml
```

## 进度条显示

执行 `optimize-plan` 时，终端会输出单行进度条，显示：

- 当前总进度百分比
- 当前阶段，例如“构建候选走廊图”“候选评估”“生成输出图层”
- 当前候选配变编号
- 局部搜索迭代进度
- 当前目标值和全局最优目标值

适合在用户规模较大、候选点较多时直观看到优化是否仍在推进，以及当前卡在哪一轮候选搜索。

## 配置说明

默认配置文件是 [configs/default_config.yaml](/C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/configs/default_config.yaml:1)。

`planning_v2` 里的关键参数包括：

- `tx_candidate_count` / `tx_prefilter_top_k`：配变候选数量与预筛数量
- `corridor_safe_margin_m`：走廊对禁区的安全边距
- `corridor_cluster_count`：用户簇数量
- `corridor_edge_max_length_m`：候选走廊边最大长度
- `build_cost_weight` / `loss_cost_weight` / `phase_unbalance_weight`：目标函数权重
- `max_service_drop_m` / `max_pole_span_m` / `voltage_drop_max_pct`：硬约束
- `alns_max_iter` / `alns_destroy_ratio`：局部重挂接改进参数
- `show_progress`：是否显示终端进度条
- `progress_bar_width`：进度条宽度

## 输出图层

标准 `GeoPackage` 图层 schema 仍由 [src/io/vector_io.py](/C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/src/io/vector_io.py:1) 维护，核心图层包括：

- `users`
- `candidate_transformer`
- `candidate_poles`
- `planned_lines`

其中 `planned_lines` 主要包含：

- `lv_line`：主干/分支低压线路
- `service_drop`：用户接户线
- `load_a_kva / load_b_kva / load_c_kva`
- `voltage_drop_pct`
- `min_clearance_m / required_clearance_m`
- `is_violation`

## 测试

运行测试：

```bash
python -m pytest -q
```

当前重点覆盖：

- 走廊图不穿越禁区
- V2 优化输出保持径向树结构
- 用户接户线长度约束
- 相别分配、线损与压降指标输出
- 终端进度条输出

## 说明

- 当前优化器是规划级近似求解器，不是施工设计级详设工具
- `radial_tree_milp.py` 现阶段使用的是不依赖外部 MILP 求解器的近似树选边方法，但接口与模块职责已经按 V2 结构拆开，后续可以在不改主流程的前提下替换为更精确的求解器
- 详细设计思路见 [PLAN_opt_V2.md](/C:/Users/LHY/Desktop/MyProject_test/DistributionNetOptimizer/PLAN_opt_V2.md:1)
