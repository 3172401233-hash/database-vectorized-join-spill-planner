# database-vectorized-join-spill-planner
规划向量化 Join 的内存预算、溢写和恢复

## 逻辑行模型

本仓库中**所有数据均为逻辑行模型**：

- 不执行真实 IO，不写真实分区文件；`spill_dir` 下的路径仅为逻辑标识。
- 内存一律以“行”为单位计量（`memory_budget_rows`），不按字节。
- 表数据用键分布（`{键: 行数}`）描述，不物化具体行。
- 溢写、刷出块、恢复配对都是对逻辑行的记账，用于验证规划正确性。

## 规划流程

1. 选行数较少的一侧为构建侧，另一侧为探测侧。
2. 单分区容量 = 内存预算 − 预留批次（探测批次 + 输出缓冲）× 批大小。
3. 构建侧按键的确定性哈希（crc32）分区；超容量分区递归再分，
   每层使用不同的哈希盐，直到 `max_depth`。
4. 热点键的所有行始终落入同一子分区，递归无法缩小；
   到达 `max_depth` 仍超容量即记为失败条件，不会无限拆分。
5. 常驻分区累计行数不超过容量，其余分区溢写到逻辑分区文件，
   并按批大小记录刷出块；规划前已刷出块原样并入日志。
6. 探测侧按与构建侧完全相同的哈希链路由，保证恢复时
   落盘分区按分区路径正确配对（`recovery_pairs()`）。

## 输入与输出

输入（`PlannerConfig`）：左右表行数、批大小、键分布、可用内存、
分区文件目录、已刷出块、最大递归深度。

输出（`SpillPlan`）：每轮内存（`rounds`）、溢写分区
（`build_partitions` / `probe_partitions`）、递归深度（分区路径长度）、
失败条件（`failures`），`render()` 可打印可读报告。

## 用法

```python
from spill_planner import PlannerConfig, SpillPlanner

dist = {"hot": 9000}
dist.update({i: 100 for i in range(10)})
cfg = PlannerConfig(
    left_rows=10000, right_rows=10000,
    left_key_dist=dict(dist), right_key_dist=dict(dist),
    batch_size=128, memory_budget_rows=2048, max_depth=2)
plan = SpillPlanner(cfg).plan()
print(plan.render())
```

## 测试

```
python -m unittest
```

覆盖均匀键、单键热点、内存临界（恰好够 / 差一行 / 不足以容纳批次）
和溢写恢复（配对、刷出块行数守恒、恢复行数守恒、确定性）。
