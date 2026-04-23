# 三维地形下的配变选址、放射式布线与三相平衡优化方案

## Summary

本阶段在既有随机地形、用户、树林、水区和人工禁区基础上，实现配变选址、动态杆塔布置、三维线路布线、ABC 相别优化、电气校核和可视化输出。

默认用户为 50 户，其中 40 户负荷为 7 kW，10 户负荷为 12 kW，功率因数均为 0.85，默认均为单相用户。单台配变固定为 630 kVA，默认总视在容量为 `(40 * 7 + 10 * 12) / 0.85 = 470.59 kVA`，容量校核可行。

优化采用“连续可行域驱动的动态布点与三维径向网络优化算法”。实现不把固定全局候选点作为唯一解空间，而是在连续可行区域中先搜索配变位置和线路走廊，再沿路径动态布置杆塔、补充接入点并生成径向网络。

树林、水区和人工划定禁区统一作为硬禁区，线路不得穿越，杆塔、配变和接户点不得落入。高压侧采用固定上级电源点到配变的放射式路径，低压侧生成以配变为根的无环径向网络。主干线路同一路径承载 `A/B/C/N` 四线，单相用户由优化算法分配接入 `A/B/C` 之一。

## Optimization Objective

核心目标是在满足容量、压降、禁区、档距、不交叉和三维地形约束的前提下，最小化建设总成本，并尽量使台区及各低压线路分段三相负载基本平衡。

推荐目标函数：

```text
minimize =
  construction_cost
  + phase_unbalance_penalty
  + voltage_violation_penalty
  + terrain_penalty
  + construction_complexity_penalty
  + infeasible_constraint_penalty
```

`construction_cost` 包括配变成本、杆塔成本、线路长度成本、导线成本、接户线成本和地形施工附加成本。`phase_unbalance_penalty` 是正式优化目标项，不是后处理，同时约束台区总三相负荷和各低压线路分段下游三相负荷。

三相不平衡指标：

```text
S_avg = (S_A + S_B + S_C) / 3
unbalance = max(|S_A - S_avg|, |S_B - S_avg|, |S_C - S_avg|) / S_avg
```

默认配置建议为 `phase_balance_target_ratio: 0.10`、`phase_balance_max_ratio: 0.15`。压降默认三相低压按 `±7%`、220 V 单相按 `+7%/-10%`，可在配置中覆盖。

## Hard Constraints

- 单台配变容量不得超过 `630 kVA * max_loading_ratio`，默认 `max_loading_ratio = 1.0`。
- 高压侧必须为从上级电源点到配变的放射式路径。
- 低压侧必须为以配变为根的无环径向网络。
- 线路不得穿越树林、水区、人工禁区和不可通行区域。
- 杆塔、配变和接户点不得落入硬禁区或不可建设区域。
- 平面投影中线路不得出现无节点交叉，允许共享端点、共杆同路径或同一路径承载 `ABCN`。
- 相邻杆塔之间的平面档距不得超过 50 m。
- 接户线平面档距不宜大于 25 m。
- 主干线路必须按同一路径承载 `A/B/C/N` 四线，不能为了相别分配拆成不同路径。
- 每个单相用户必须由优化算法写入 `assigned_phase in {A, B, C}`。
- 每条低压线路段必须汇总下游 `A/B/C` 负荷、零线电流估算、三相不平衡率和压降。
- 杆塔和配变位置必须满足局部坡度、粗糙度和可施工阈值。
- 线路边必须基于 DTM 剖面校核地形净距；初版使用简化高程采样，后续可扩展弧垂模型。

## Data And Interfaces

用户生成配置采用 `load_groups`：

```yaml
users:
  count: 50
  load_groups:
    - count: 40
      load_kw: 7.0
      power_factor: 0.85
      phase_type: single
    - count: 10
      load_kw: 12.0
      power_factor: 0.85
      phase_type: single
```

保留未来随机接口：`load_kw_range`、`power_factor_range`、`phase_type_distribution`。默认不启用随机负荷和随机相别。

`users` 图层新增或预留字段：`load_kw`、`power_factor`、`phase_type`、`assigned_phase`、`apparent_kva`、`elev_m`、`connected_node_id`、`voltage_drop_pct`。

`candidate_transformer` 图层字段：`transformer_id`、`candidate_id`、`capacity_kva`、`fixed_cost`、`elev_m`、`ground_slope_deg`、`buildable_score`、`source`。

`candidate_poles` 图层字段：`pole_id`、`candidate_id`、`pole_type`、`pole_height_m`、`fixed_cost`、`elev_m`、`ground_slope_deg`、`source`。

`planned_lines` 图层字段：`line_id`、`line_type`、`from_node`、`to_node`、`phase_set`、`service_phase`、`horizontal_length_m`、`length_3d_m`、`dz_m`、`slope_deg`、`cost`、`load_a_kva`、`load_b_kva`、`load_c_kva`、`neutral_current_a`、`voltage_drop_pct`、`is_violation`。

优化汇总输出为 `data/outputs/plans/optimization_summary.json`，记录总成本、配变位置、高程、线路总三维长度、杆塔数量、最大高差、最大坡度、最大压降、三相负荷、不平衡率和不可行原因。

## Three-Dimensional Modeling

所有用户、配变、杆塔和上级电源点均应有三维坐标。若配置未给定高程，则从 `DTM` 按坐标采样。

线路边同时保存平面长度和三维长度：

```text
horizontal_length = sqrt(dx^2 + dy^2)
length_3d = sqrt(dx^2 + dy^2 + dz^2)
```

杆塔档距和接户线档距按平面长度校核；线路材料成本、压降计算和施工成本默认按三维长度计算。路径搜索基于综合代价场，不只按平面最短距离，代价包含平面长度、三维长度、高差、坡度、地表粗糙度、靠近禁区边界风险和净距风险。

3D 图形中线路不贴地绘制，应按杆塔高度抬升后连接，体现真实空间关系。

## Algorithm Plan

V1 采用“连续可行域驱动的动态布点与三维径向网络优化算法”：先完成配变选址与线路走廊搜索，再在局部走廊内动态生成杆塔和连接关系，最终形成满足地形、障碍、电气与相别平衡约束的规划方案。

1. 硬禁区统一建模：将树林、水区及人工禁区合并为不可穿越、不可落点的约束掩膜。若后续需要表达施工困难区域，可额外定义软约束区并通过成本惩罚参与优化。
2. 连续可行域构建：在硬禁区之外，结合坡度、粗糙度和局部可施工性构建连续可行域。不在全局均匀撒布密集候选点，只确定配变搜索区域和潜在线路走廊范围。
3. 配变位置搜索：在连续可行域内搜索配变位置，采样高程并计算坡度、粗糙度和可施工性。V1 可采用粗网格筛选加局部细化搜索。
4. 三维路径走廊生成：在配变位置基础上生成高压侧和低压侧三维路径走廊。路径搜索基于综合代价场，必须绕开树林、水区和人工禁区。
5. 沿走廊动态布置杆塔：依据最大平面档距 50 m、地形高差、局部坡度和净距要求自动布置杆塔；对转折点、净距不足点、坡度突变点和重要接入点附近杆位进行动态增删和微调。
6. 用户接入与局部补杆：对每个用户在最大接户线平面距离 25 m 内优先搜索可接入杆塔；若现有杆塔无法满足，允许在局部路径走廊内增设接入杆塔；若仍不可行，触发局部路径重规划或输出诊断。
7. 低压径向网络生成：将各用户接入路径叠加为局部连接图，在满足放射式、无环、共享主干优先和总成本最小的原则下，生成以配变为根的低压径向网络。
8. ABC 相别优化：在低压径向网络确定后，对所有单相用户执行 `A/B/C` 相别分配。初始分配按视在容量从大到小贪心分配到累计负荷最轻相，再通过局部交换、支路内调整及相邻支路协调降低不平衡。
9. 电气校核与修复：依据径向网络结构和 `assigned_phase` 自下而上汇总每条线路段的 `A/B/C` 负荷、零线电流、不平衡率及末端压降。若出现压降超限、零线电流过大或相间不平衡超限，触发相别重分配、局部换接、局部增杆、局部路径替换或共享主干调整。
10. 方案评估与输出：在所有可行方案中选择目标函数最小的结果，输出规划图层、优化汇总 JSON、2D 静态图、3D 静态图和 3D 动态交互图。
11. 不可行诊断输出：若任意硬约束无法满足，优化器不得静默失败，应明确输出不可行原因、约束冲突位置及诊断图。

## Alternative Algorithms

- 连续优化 + 路径规划：在连续可行域中做配变位置搜索，再用 A*/Theta*/Visibility Graph 生成避障路径。
- PRM/RRT*：适合连续空间避障与复杂禁区绕行，可结合后插杆和电气校核。
- MILP/MIQP：将配变位置、路径边、杆塔点、用户接入、相别分配和三维约束统一建模，小规模可求较优或最优，但需要求解器。
- 遗传算法、模拟退火、禁忌搜索、NSGA-II：适合复杂非线性成本、高差惩罚和杆塔型号选择，但调参成本较高。
- K-shortest-paths 重布线：为关键线路生成多条三维候选路径，通过局部替换解决交叉、净距、压降、三相平衡和成本冲突。

## Visualization Plan

- `data/outputs/plots/optimized_plan_2d.png`：2D 静态图，叠加地形、树林、水区、禁区、用户、配变、杆塔、高压线、低压主干线和接户线。
- `data/outputs/plots/optimized_plan_3d_static.png`：3D 静态图，基于 DTM 曲面展示真实高程、配变、杆塔和线路。
- `data/outputs/plots/optimized_plan_3d_dynamic.html`：3D 动态交互图，支持旋转、缩放、图层开关和 hover 信息。

用户按 `A/B/C` 相别使用固定颜色，接户线按接入相着色，主干线标注 `ABCN`。hover 信息包括用户负荷、功率因数、接入相、高程、线路平面长度、三维长度、高差、下游三相负荷、压降和净距状态。

## Acceptance Tests

- 默认用户生成：50 户，其中 40 户 7 kW、10 户 12 kW，功率因数均为 0.85，默认单相，且每户有 `elev_m`。
- 硬禁区：树林、水区、人工禁区均不可穿越，杆塔、配变和线路不得落入或穿越这些区域。
- 容量校验：默认总视在容量约 470.59 kVA，单台 630 kVA 判定可行；超过容量时返回不可行原因。
- 动态接户覆盖：每个用户 25 m 范围内应存在可行接户点，否则触发局部补点或输出覆盖诊断。
- 路径后插杆：沿路径布置杆塔后，相邻杆塔平面距离不得超过 50 m。
- 三维边计算：`length_3d >= horizontal_length`，高差、坡度、DTM 剖面采样和净距校核正确。
- ABC 相别分配：所有单相用户都有 `assigned_phase in {A, B, C}`，台区总三相负荷不平衡率低于配置阈值。
- 可视化输出：2D 静态图、3D 静态图和 3D 动态图均生成。

## Assumptions

本阶段固定单台 630 kVA 配变；后续再扩展为多台、多容量型号自动选择。上级高压电源点由配置给定，若未给定高程则从 `DTM` 采样。初版导线弧垂和杆塔结构只做简化净距校核，更精细的力学弧垂、转角杆、耐张杆和跨越规范作为后续扩展。当前方案为规划辅助工具，正式工程设计仍需按现行标准、当地供电公司要求和实际导线、杆塔型号复核。
