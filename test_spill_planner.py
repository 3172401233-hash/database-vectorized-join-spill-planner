import unittest

from spill_planner import FlushBlock, PlannerConfig, SpillPlanner


def uniform_dist(num_keys, rows_per_key):
    return {i: rows_per_key for i in range(num_keys)}


class UniformKeyTest(unittest.TestCase):
    """均匀键：分区均衡，规划成功，每轮内存不超预算。"""

    def test_uniform_keys(self):
        dist = uniform_dist(100, 100)  # 10000 行
        cfg = PlannerConfig(
            left_rows=10000, right_rows=10000,
            left_key_dist=dict(dist), right_key_dist=dict(dist),
            batch_size=128, memory_budget_rows=4096, max_depth=3)
        plan = SpillPlanner(cfg).plan()
        self.assertTrue(plan.ok)
        self.assertEqual(plan.build_side, "left")
        for rnd in plan.rounds:
            self.assertLessEqual(rnd.memory_rows, rnd.memory_budget_rows)
            self.assertEqual(rnd.status, "ok")
        self.assertEqual(
            sum(p.rows for p in plan.build_partitions), 10000)
        self.assertEqual(
            sum(p.rows for p in plan.probe_partitions), 10000)
        # 容量 4096-256=3840，10000 行初始分 3 区；哈希偏斜可能触发再拆分
        self.assertEqual(plan.num_partitions, 3)
        spilled = [p for p in plan.build_partitions if p.spilled]
        resident = [p for p in plan.build_partitions if p.resident]
        self.assertTrue(spilled)
        self.assertTrue(resident)
        for part in plan.build_partitions:
            self.assertLessEqual(part.rows, plan.capacity_rows)
        self.assertLessEqual(sum(p.rows for p in resident),
                             plan.capacity_rows)


class HotKeyTest(unittest.TestCase):
    """单键热点：递归拆分无法缩小热点分区，到达最大深度后报失败。"""

    def test_single_hot_key_fails_at_max_depth(self):
        dist = {"hot": 9000}
        dist.update(uniform_dist(10, 100))  # 共 10000 行
        cfg = PlannerConfig(
            left_rows=10000, right_rows=10000,
            left_key_dist=dict(dist), right_key_dist=dict(dist),
            batch_size=128, memory_budget_rows=2048, max_depth=2)
        plan = SpillPlanner(cfg).plan()
        self.assertFalse(plan.ok)
        self.assertEqual(len(plan.failures), 1)
        failure = plan.failures[0]
        self.assertEqual(failure.depth, 2)
        self.assertEqual(failure.dominant_key, "hot")
        self.assertEqual(failure.rows, 9000)
        self.assertGreater(failure.rows, failure.capacity_rows)
        self.assertIn("最大递归深度", failure.reason)
        # 失败分区既不常驻也不溢写，对应轮次标记为 failed
        failed_rounds = [r for r in plan.rounds if r.status == "failed"]
        self.assertEqual(len(failed_rounds), 1)
        self.assertGreater(failed_rounds[0].memory_rows,
                           failed_rounds[0].memory_budget_rows)
        # 其余分区仍正常规划
        ok_rounds = [r for r in plan.rounds if r.status == "ok"]
        self.assertTrue(ok_rounds)
        for rnd in ok_rounds:
            self.assertLessEqual(rnd.memory_rows, rnd.memory_budget_rows)

    def test_hot_key_recursion_is_bounded(self):
        dist = {"hot": 5000}
        cfg = PlannerConfig(
            left_rows=5000, right_rows=5000,
            left_key_dist=dict(dist), right_key_dist=dict(dist),
            batch_size=64, memory_budget_rows=1024, max_depth=3)
        plan = SpillPlanner(cfg).plan()
        self.assertFalse(plan.ok)
        # 递归被 max_depth 截断，不会无限拆分
        self.assertTrue(all(f.depth <= 3 for f in plan.failures))
        self.assertTrue(all(r.depth <= 3 for r in plan.rounds))


class MemoryEdgeTest(unittest.TestCase):
    """内存临界：预算恰好够 / 差一行 / 不足以容纳批次。"""

    def setUp(self):
        self.dist = uniform_dist(10, 100)  # 1000 行

    def make_config(self, budget):
        return PlannerConfig(
            left_rows=1000, right_rows=1000,
            left_key_dist=dict(self.dist), right_key_dist=dict(self.dist),
            batch_size=100, memory_budget_rows=budget, max_depth=3)

    def test_budget_exactly_fits(self):
        # 容量 = 1200 - 2*100 = 1000，恰好容纳全部构建行
        plan = SpillPlanner(self.make_config(1200)).plan()
        self.assertTrue(plan.ok)
        self.assertEqual(len(plan.rounds), 1)
        self.assertEqual(plan.rounds[0].memory_rows, 1200)
        self.assertFalse(any(p.spilled for p in plan.build_partitions))

    def test_budget_one_row_short(self):
        # 容量 = 1199 - 200 = 999 < 1000，必须拆分并溢写
        plan = SpillPlanner(self.make_config(1199)).plan()
        self.assertTrue(plan.ok)
        self.assertGreater(len(plan.build_partitions), 1)
        self.assertTrue(any(p.spilled for p in plan.build_partitions))
        for rnd in plan.rounds:
            self.assertLessEqual(rnd.memory_rows, 1199)

    def test_budget_below_batch_reserve(self):
        # 预算连探测批次+输出缓冲都不够
        plan = SpillPlanner(self.make_config(199)).plan()
        self.assertFalse(plan.ok)
        self.assertEqual(len(plan.rounds), 0)
        self.assertIn("内存预算不足", plan.failures[0].reason)


class SpillRecoveryTest(unittest.TestCase):
    """溢写恢复：落盘分区按路径配对，刷出块行数守恒。"""

    def setUp(self):
        dist = uniform_dist(100, 100)  # 10000 行
        self.existing = [FlushBlock(
            side="build", path=(0,), block_index=0, rows=64,
            file_path="spill/build_d1_p0.spill")]
        cfg = PlannerConfig(
            left_rows=10000, right_rows=10000,
            left_key_dist=dict(dist), right_key_dist=dict(dist),
            batch_size=128, memory_budget_rows=2048, max_depth=3,
            flushed_blocks=list(self.existing))
        self.plan = SpillPlanner(cfg).plan()

    def test_recovery_pairs_match_by_path(self):
        self.assertTrue(self.plan.ok)
        pairs = self.plan.recovery_pairs()
        spilled_build = [p for p in self.plan.build_partitions if p.spilled]
        self.assertEqual(len(pairs), len(spilled_build))
        self.assertTrue(pairs)
        probe_by_path = {p.path: p for p in self.plan.probe_partitions}
        for pair in pairs:
            self.assertIn("build", pair.build_file)
            probe = probe_by_path[pair.path]
            self.assertTrue(probe.spilled)
            self.assertEqual(pair.probe_file, probe.file_path)
            self.assertIn("probe", pair.probe_file)

    def test_flushed_blocks_conserve_rows(self):
        # 新刷出块（不含规划前已存在的块）覆盖所有溢写分区
        new_blocks = self.plan.flushed_blocks[len(self.existing):]
        for side, partitions in (("build", self.plan.build_partitions),
                                 ("probe", self.plan.probe_partitions)):
            for part in partitions:
                blocks = [b for b in new_blocks
                          if b.side == side and b.path == part.path]
                if part.spilled:
                    self.assertEqual(sum(b.rows for b in blocks), part.rows)
                    self.assertEqual(len(blocks), part.blocks)
                    self.assertTrue(all(b.rows <= 128 for b in blocks))
                else:
                    self.assertEqual(blocks, [])
        # 规划前已刷出的块原样保留
        self.assertEqual(self.plan.flushed_blocks[:len(self.existing)],
                         self.existing)

    def test_restore_row_conservation(self):
        # 模拟恢复：常驻 + 配对文件中的行数等于原始行数
        build_resident = sum(p.rows for p in self.plan.build_partitions
                             if p.resident)
        probe_resident = sum(p.rows for p in self.plan.probe_partitions
                             if p.resident)
        pairs = self.plan.recovery_pairs()
        build_restored = build_resident + sum(p.build_rows for p in pairs)
        probe_restored = probe_resident + sum(p.probe_rows for p in pairs)
        self.assertEqual(build_restored, 10000)
        self.assertEqual(probe_restored, 10000)

    def test_plan_is_deterministic(self):
        dist = uniform_dist(100, 100)
        cfg = PlannerConfig(
            left_rows=10000, right_rows=10000,
            left_key_dist=dict(dist), right_key_dist=dict(dist),
            batch_size=128, memory_budget_rows=2048, max_depth=3)
        again = SpillPlanner(cfg).plan()
        self.assertEqual(
            [p.file_path for p in self.plan.build_partitions],
            [p.file_path for p in again.build_partitions])
        self.assertEqual(
            [(r.path, r.memory_rows) for r in self.plan.rounds],
            [(r.path, r.memory_rows) for r in again.rounds])


if __name__ == "__main__":
    unittest.main()
