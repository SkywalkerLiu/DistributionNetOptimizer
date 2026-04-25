# DistributionNetOptimizer

用于配电网规划前期场景构建与 V2 低压配电优化的 Python 项目。

当前版本已经按 `PLAN_opt_V2.md` 完成优化器主架构重构，旧的“连续域路径搜索 + 可见图 + 路径后插杆”不再参与执行链路。项目现在统一采用：

`候选走廊图 -> 径向树选边 -> 三相分配 -> 功率流评估 -> 杆塔派生`

## 核心能力

- 生成地形、坡度、粗糙度、可建设区与禁建区
- 生成用户、树林、水域、人工禁建区等场景要素
- 基于候选走廊图构建工程可行的低压布线候选网络
- 在走廊图上选择配变位置、共享主干与分支树结构
- 对单相用户进行 `A/B/C` 相别优化
- 计算线路电流、线损、压降和相不平衡指标
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
  radial_tree_milp.py        # Pyomo + HiGHS 径向树 MILP 求解
  phase_assignment.py        # 三相分配优化
  bfs_power_flow.py          # 相负荷汇总、线路电流、线损与压降评估
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

默认优化求解器为开源组合 `pyomo` + `highspy`。项目当前只接入推荐的 HiGHS 后端，不包含 OR-Tools、Gurobi 或 SCIP 后端。

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
- `build_cost_weight` / `phase_unbalance_weight`：目标函数权重
- `max_service_drop_m` / `max_pole_span_m`：接户线长度和杆距硬约束；`voltage_drop_max_pct` 仅作为旧配置兼容的压降参考值
- `pole_user_clearance_m`：杆塔/配变候选点到用户点的最小水平净距，默认 `5.0m`
- `line_user_clearance_m`：公共低压线路到用户点的最小水平净距，默认 `1.0m`；接户线允许连接自己的用户点，但仍会避让其他用户点
- `solver_backend`：当前仅支持 `highs`
- `milp_time_limit_s` / `mip_gap`：HiGHS 求解时间限制与 MIP gap
- `parallel_candidate_eval`：是否并行评估配变候选，默认 `false`
- `parallel_workers`：候选评估进程数，默认 `1`
- `highs_threads_per_worker`：每个 worker 内 HiGHS 使用的线程数，默认 `1`
- `local_search_top_k`：只对初始评估排名前 K 的候选做局部重挂接，默认 `1`
- `emit_performance_metrics`：是否在 summary 中输出性能指标，默认 `true`
- `alns_max_iter` / `alns_destroy_ratio`：局部重挂接改进参数
- `show_progress`：是否显示终端进度条
- `progress_bar_width`：进度条宽度

### 路径长度与多方向出线控制

- `path_length_penalty_weight`：对用户到配变供电路径长度的惩罚权重。
- `max_user_path_length_m`：用户到配变主干路径长度的建议上限，超过后触发二次惩罚。
- `max_user_path_penalty_weight`：超过最大供电路径长度后的惩罚权重。
- `load_weighted_path_penalty_weight`：负荷加权路径长度惩罚权重。
- `root_feeder_min_count`：鼓励配变至少向多少个方向出线。
- `root_feeder_count_penalty_weight`：根节点出线数量不足时的惩罚权重。
- `corridor_neighbor_count`：每个走廊节点连接的近邻数量，与候选方案池大小无关。

## 性能优化与运行时间调优

V2 优化的主要耗时来自多次构建 Pyomo 模型并调用 HiGHS 求解径向树 MILP。Pyomo 建模和单个 HiGHS 求解通常不会让整机 CPU 长时间满载，所以看到 CPU 总占用低于 `20%` 不一定代表程序卡死；更常见的原因是候选配变串行评估和重复 MILP 调用。

当前默认采用“质量优先”的两阶段搜索：

1. 并行评估全部初始配变候选。
2. 按初始目标值排序，只对前 `local_search_top_k` 个候选做局部重挂接。
3. 最终从全部初始解和 top-k 改进解中选最优解。

推荐配置如下：

```yaml
planning_v2:
  parallel_candidate_eval: false
  parallel_workers: 1
  highs_threads_per_worker: 1
  local_search_top_k: 1
  emit_performance_metrics: true
```

调参建议：

- 质量优先：提高 `tx_prefilter_top_k`、降低 `mip_gap`，并按本机权限情况打开并行和增大 `local_search_top_k`。
- 更快出图：降低 `local_search_top_k` 或 `alns_max_iter`，例如 `local_search_top_k: 4`、`alns_max_iter: 100`。
- 调试复现：设置 `parallel_workers: 1` 或 `parallel_candidate_eval: false`，让候选评估串行执行。
- 多进程受限环境：建议保持默认串行配置，避免 Windows sandbox 或权限策略阻止创建进程池。

### V2 线损与压降处理

V2 保留线路阻抗、负载电流、线损和压降计算。

- 线损以有功功率损耗 `kW` 表示；
- 线损通过 `loss_kw_weight` 进入优化目标；
- 不使用电价或运行小时折算线损；
- 压降不作为硬约束；
- 压降超限不会导致 `infeasible`；
- 压降通过 `voltage_drop_penalty_weight` 作为软惩罚进入目标函数；
- summary 中输出压降较大的用户；
- summary 中输出台区电流最大的线路段及其 A/B/C 三相电流；
- 接户线压降不单独记录。

执行完成后，可在 `data/outputs/plans/optimization_summary.json` 的 `performance` 字段查看阶段耗时和 MILP 调用次数，例如：

```json
{
  "performance": {
    "total_duration_s": 412.8,
    "corridor_build_duration_s": 8.2,
    "initial_candidate_eval_duration_s": 146.5,
    "local_search_duration_s": 231.4,
    "worker_count": 6,
    "milp_solve_count": 96,
    "milp_cache_hit_count": 12,
    "slowest_candidate_duration_s": 52.3
  }
}
```

优化 summary 会输出稳定的不可行原因代码，例如：

```yaml
infeasible_reason:
  - service_drop_too_long
  - no_corridor_to_user_17
  - phase_unbalance_exceeded
```

当前已覆盖的主要 reason code 包括：

- `service_drop_too_long`
- `phase_unbalance_exceeded`
- `transformer_overloaded`
- `line_vertical_clearance_exceeded`
- `line_user_clearance_exceeded`
- `pole_user_clearance_exceeded`
- `radial_tree_infeasible`
- `no_corridor_to_user_<user_id>`

## 输出图层

标准 `GeoPackage` 图层 schema 仍由 [src/io/vector_io.py](src/io/vector_io.py) 维护，核心图层包括：

- `users`
- `candidate_transformer`
- `candidate_poles`
- `planned_lines`

其中 `planned_lines` 主要包含：

- `lv_line`：主干/分支低压线路
- `service_drop`：用户接户线
- `load_a_kva / load_b_kva / load_c_kva`
- `voltage_drop_pct`：线路和用户侧压降诊断值
- `min_clearance_m / required_clearance_m`
- `user_clearance_m / required_user_clearance_m`
- `violation_reason`
- `is_violation`

## 测试

运行测试：

```bash
python -m pytest -q
```

当前重点覆盖：

- 走廊图不穿越禁区
- 杆塔水平避让用户 `5m`，公共线路水平避让用户 `1m`
- V2 优化输出保持径向树结构
- 用户接户线长度约束
- 相别分配、线损和压降诊断指标输出
- 不可行 reason code 输出
- 两阶段候选评估、性能指标和并行 worker 配置
- 终端进度条输出

## 说明

- 当前优化器是规划级近似求解器，不是施工设计级详设工具
- `radial_tree_milp.py` 使用 Pyomo 建模并调用 HiGHS 求解径向树选边 MILP；当前项目只维护这一推荐开源求解器后端
- 详细设计思路见 [PLAN_opt_V2.md](PLAN_opt_V2.md)
