"""向量化 Hash Join 溢写规划器（逻辑行模型）。

所有数据均为逻辑行：不执行真实 IO，内存以“行”为单位计量，
分区文件仅为逻辑路径，用于规划落盘与恢复时的配对。
"""
from __future__ import annotations

import math
import os
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

KeyDist = Dict[object, int]
Path = Tuple[int, ...]


def _hash_key(key: object, salt: str) -> int:
    """确定性哈希（跨进程稳定），用于分区路由。"""
    return zlib.crc32("{}|{}".format(salt, key).encode("utf-8"))


def _normalize_dist(dist: Optional[KeyDist], rows: int) -> KeyDist:
    """校验或合成键分布。未提供时按均匀分布合成。"""
    if rows < 0:
        raise ValueError("行数不能为负")
    if dist is None:
        if rows == 0:
            return {}
        num_keys = max(1, min(rows, 100))
        base, rem = divmod(rows, num_keys)
        return {i: base + (1 if i < rem else 0) for i in range(num_keys)}
    total = 0
    for key, cnt in dist.items():
        if not isinstance(cnt, int) or cnt < 0:
            raise ValueError("键分布计数必须为非负整数: {!r}".format(key))
        total += cnt
    if total != rows:
        raise ValueError(
            "键分布总行数 {} 与声明行数 {} 不一致".format(total, rows))
    return dict(dist)


@dataclass
class FlushBlock:
    """已刷出块：一次落盘的逻辑记录。"""
    side: str
    path: Path
    block_index: int
    rows: int
    file_path: str


@dataclass
class PartitionRecord:
    side: str
    path: Path
    depth: int
    rows: int
    resident: bool
    spilled: bool
    paired: bool
    file_path: Optional[str]
    blocks: int


@dataclass
class RoundRecord:
    """一轮构建+探测的内存与溢写情况。"""
    round_id: int
    path: Path
    depth: int
    build_rows: int
    probe_rows: int
    probe_batches: int
    memory_rows: int
    memory_budget_rows: int
    spilled_files: Tuple[str, ...]
    status: str  # "ok" / "failed"


@dataclass
class FailureRecord:
    path: Path
    depth: int
    rows: int
    capacity_rows: int
    dominant_key: object
    reason: str


@dataclass
class RecoveryPair:
    """恢复时配对的构建/探测分区文件。"""
    path: Path
    build_file: str
    probe_file: Optional[str]
    build_rows: int
    probe_rows: int


@dataclass
class PlannerConfig:
    left_rows: int
    right_rows: int
    batch_size: int
    memory_budget_rows: int
    left_key_dist: Optional[KeyDist] = None
    right_key_dist: Optional[KeyDist] = None
    num_partitions: int = 0       # 0 = 按容量自动推导
    max_depth: int = 3            # 热点键递归拆分的最大深度
    spill_dir: str = "spill"      # 分区文件的逻辑目录
    reserve_batches: int = 2      # 预留：探测批次 + 输出缓冲
    flushed_blocks: List[FlushBlock] = field(default_factory=list)


@dataclass
class SpillPlan:
    build_side: str
    probe_side: str
    capacity_rows: int
    num_partitions: int
    max_depth: int
    rounds: List[RoundRecord]
    build_partitions: List[PartitionRecord]
    probe_partitions: List[PartitionRecord]
    flushed_blocks: List[FlushBlock]
    failures: List[FailureRecord]

    @property
    def ok(self) -> bool:
        return not self.failures

    def recovery_pairs(self) -> List[RecoveryPair]:
        """恢复时按分区路径配对落盘的构建/探测文件。"""
        probe_by_path = {p.path: p for p in self.probe_partitions}
        pairs = []
        for bp in self.build_partitions:
            if not bp.spilled:
                continue
            pp = probe_by_path.get(bp.path)
            pairs.append(RecoveryPair(
                path=bp.path,
                build_file=bp.file_path or "",
                probe_file=pp.file_path if pp else None,
                build_rows=bp.rows,
                probe_rows=pp.rows if pp else 0,
            ))
        return pairs

    def render(self) -> str:
        lines = [
            "构建侧={} 探测侧={} 单分区容量={} 行 分区数={} 最大递归深度={}".format(
                self.build_side, self.probe_side, self.capacity_rows,
                self.num_partitions, self.max_depth),
        ]
        for r in self.rounds:
            files = ", ".join(r.spilled_files) if r.spilled_files else "无"
            lines.append(
                "轮次 {}: 分区 {} 深度 {} 构建 {} 行 探测 {} 行 ({} 批) "
                "内存 {}/{} 行 状态 {} 溢写 [{}]".format(
                    r.round_id, _fmt_path(r.path), r.depth, r.build_rows,
                    r.probe_rows, r.probe_batches, r.memory_rows,
                    r.memory_budget_rows, r.status, files))
        for f in self.failures:
            lines.append(
                "失败: 分区 {} 深度 {} 行数 {} > 容量 {} 主导键 {!r}: {}".format(
                    _fmt_path(f.path), f.depth, f.rows, f.capacity_rows,
                    f.dominant_key, f.reason))
        if self.ok:
            lines.append("结果: 规划成功")
        return "\n".join(lines)


def _fmt_path(path: Path) -> str:
    return "(" + ",".join(str(p) for p in path) + ")"


class SpillPlanner:
    """规划构建侧分区、溢写与探测侧读取。"""

    def __init__(self, config: PlannerConfig):
        if config.batch_size <= 0:
            raise ValueError("批大小必须为正")
        if config.memory_budget_rows <= 0:
            raise ValueError("内存预算必须为正")
        if config.max_depth < 0:
            raise ValueError("最大递归深度不能为负")
        self.config = config
        self._splits: Dict[Path, int] = {}
        self._capacity = 0
        self._num_partitions = 1

    def plan(self) -> SpillPlan:
        cfg = self.config
        left_dist = _normalize_dist(cfg.left_key_dist, cfg.left_rows)
        right_dist = _normalize_dist(cfg.right_key_dist, cfg.right_rows)
        if cfg.left_rows <= cfg.right_rows:
            build_side, probe_side = "left", "right"
            build_rows, build_dist = cfg.left_rows, left_dist
            probe_dist = right_dist
        else:
            build_side, probe_side = "right", "left"
            build_rows, build_dist = cfg.right_rows, right_dist
            probe_dist = left_dist

        capacity = cfg.memory_budget_rows - cfg.reserve_batches * cfg.batch_size
        self._capacity = capacity
        failures: List[FailureRecord] = []
        if capacity <= 0:
            failures.append(FailureRecord(
                path=(), depth=0, rows=build_rows, capacity_rows=capacity,
                dominant_key=None,
                reason="内存预算不足以容纳探测批次与输出缓冲"))
            return SpillPlan(
                build_side=build_side, probe_side=probe_side,
                capacity_rows=capacity, num_partitions=0,
                max_depth=cfg.max_depth, rounds=[], build_partitions=[],
                probe_partitions=[], flushed_blocks=list(cfg.flushed_blocks),
                failures=failures)

        num_partitions = cfg.num_partitions or max(
            1, math.ceil(build_rows / capacity))
        self._num_partitions = num_partitions

        leaves: List[Tuple[Path, KeyDist, int]] = []
        self._split_build(build_dist, (), leaves, failures)
        leaves.sort(key=lambda leaf: leaf[0])

        probe_buckets: Dict[Path, int] = {}
        for key, cnt in probe_dist.items():
            path = self._route(key)
            probe_buckets[path] = probe_buckets.get(path, 0) + cnt

        flushed = list(cfg.flushed_blocks)
        build_partitions: List[PartitionRecord] = []
        probe_partitions: List[PartitionRecord] = []
        rounds: List[RoundRecord] = []

        used = 0
        spill_state: Dict[Path, Tuple[bool, bool]] = {}
        for path, _dist, rows in leaves:
            failed = rows > capacity
            if failed:
                resident, spilled = False, False
            elif used + rows <= capacity:
                resident, spilled = True, False
                used += rows
            else:
                resident, spilled = False, True
            spill_state[path] = (resident, spilled)
            file_path = self._file("build", path) if spilled else None
            blocks = 0
            if spilled:
                blocks = self._flush(flushed, "build", path, rows, file_path)
            build_partitions.append(PartitionRecord(
                side="build", path=path, depth=len(path), rows=rows,
                resident=resident, spilled=spilled, paired=True,
                file_path=file_path, blocks=blocks))

        build_leaf_rows = {path: rows for path, _d, rows in leaves}
        for path in sorted(probe_buckets):
            rows = probe_buckets[path]
            paired = path in spill_state
            resident = spilled = False
            file_path = None
            blocks = 0
            if paired:
                b_resident, b_spilled = spill_state[path]
                failed = build_leaf_rows[path] > capacity
                if b_resident:
                    resident = True
                if b_spilled or failed:
                    spilled = True
                    file_path = self._file("probe", path)
                    blocks = self._flush(
                        flushed, "probe", path, rows, file_path)
            probe_partitions.append(PartitionRecord(
                side="probe", path=path, depth=len(path), rows=rows,
                resident=resident, spilled=spilled, paired=paired,
                file_path=file_path, blocks=blocks))

        probe_file_by_path = {p.path: p.file_path for p in probe_partitions}
        for rec in build_partitions:
            probe_rows = probe_buckets.get(rec.path, 0)
            probe_batch = min(cfg.batch_size, probe_rows)
            memory = rec.rows + probe_batch + probe_batch
            files = []
            if rec.file_path:
                files.append(rec.file_path)
            pfile = probe_file_by_path.get(rec.path)
            if pfile:
                files.append(pfile)
            rounds.append(RoundRecord(
                round_id=len(rounds), path=rec.path, depth=rec.depth,
                build_rows=rec.rows, probe_rows=probe_rows,
                probe_batches=math.ceil(probe_rows / cfg.batch_size),
                memory_rows=memory,
                memory_budget_rows=cfg.memory_budget_rows,
                spilled_files=tuple(files),
                status="failed" if rec.rows > capacity else "ok"))

        return SpillPlan(
            build_side=build_side, probe_side=probe_side,
            capacity_rows=capacity, num_partitions=num_partitions,
            max_depth=cfg.max_depth, rounds=rounds,
            build_partitions=build_partitions,
            probe_partitions=probe_partitions,
            flushed_blocks=flushed, failures=failures)

    def _salt(self, depth: int) -> str:
        return "d{}".format(depth)

    def _file(self, side: str, path: Path) -> str:
        name = "{}_d{}_p{}.spill".format(
            side, len(path), "_".join(str(p) for p in path) or "root")
        return os.path.join(self.config.spill_dir, name)

    def _flush(self, flushed: List[FlushBlock], side: str, path: Path,
               rows: int, file_path: str) -> int:
        blocks = math.ceil(rows / self.config.batch_size)
        for idx in range(blocks):
            block_rows = min(self.config.batch_size,
                             rows - idx * self.config.batch_size)
            flushed.append(FlushBlock(
                side=side, path=path, block_index=idx,
                rows=block_rows, file_path=file_path))
        return blocks

    def _split_build(self, dist: KeyDist, path: Path,
                     leaves: List[Tuple[Path, KeyDist, int]],
                     failures: List[FailureRecord]) -> None:
        rows = sum(dist.values())
        depth = len(path)
        if rows <= self._capacity:
            leaves.append((path, dist, rows))
            return
        if depth >= self.config.max_depth:
            leaves.append((path, dist, rows))
            dominant = None
            if dist:
                dominant = max(dist.items(), key=lambda kv: kv[1])[0]
            failures.append(FailureRecord(
                path=path, depth=depth, rows=rows,
                capacity_rows=self._capacity, dominant_key=dominant,
                reason="热点键无法继续拆分：已达最大递归深度 {}".format(
                    self.config.max_depth)))
            return
        fanout = self._num_partitions
        self._splits[path] = fanout
        buckets: List[KeyDist] = [dict() for _ in range(fanout)]
        salt = self._salt(depth)
        for key, cnt in dist.items():
            buckets[_hash_key(key, salt) % fanout][key] = cnt
        for idx, sub in enumerate(buckets):
            if sub:
                self._split_build(sub, path + (idx,), leaves, failures)

    def _route(self, key: object) -> Path:
        """探测侧按与构建侧相同的哈希链路由，保证恢复时正确配对。"""
        path: Path = ()
        while path in self._splits:
            fanout = self._splits[path]
            idx = _hash_key(key, self._salt(len(path))) % fanout
            path = path + (idx,)
        return path
