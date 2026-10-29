"""python -m unittest scripts/vector-index/test_embed_store_mlx.py  (needs numpy only)"""
import os, sys, unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embed_store_mlx as E


def unit(n):
    v = np.random.default_rng(0).normal(size=(n, E.DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class DocText(unittest.TestCase):
    def test_matches_the_worker(self):
        self.assertEqual(E.doc_text("T", " S ", "N", "X"), "T\nS\nN")
        self.assertEqual(E.doc_text("T", None, None, "X"), "T\nX")
        self.assertEqual(E.doc_text(" ", None, "", None), "")
        self.assertEqual(len(E.doc_text("a" * 40000, None, None, None)), 32 * 1024)


class Shards(unittest.TestCase):
    def test_plan_covers_every_id_once_in_order(self):
        shards = E.plan_shards(list(range(10)), 4)
        self.assertEqual([s["name"] for s in shards], ["shard-00000", "shard-00001", "shard-00002"])
        self.assertEqual([i for s in shards for i in s["ids"]], list(range(10)))

    def test_resume_skips_finished_shards(self):
        shards = E.plan_shards(list(range(10)), 4)
        self.assertEqual([s["name"] for s in E.pending(shards, {"shard-00001"})], ["shard-00000", "shard-00002"])


class Checks(unittest.TestCase):
    def test_accepts_unit_vectors(self):
        self.assertEqual(E.problems(unit(3), 3), [])

    def test_rejects_wrong_shape_nan_and_length(self):
        self.assertTrue(E.problems(unit(2), 3)[0].startswith("shape"))
        bad = unit(3); bad[1, 0] = np.nan
        self.assertEqual(E.problems(bad, 3), ["values not finite"])
        self.assertTrue(E.problems(unit(3) * 2, 3)[0].startswith("norm"))


class Assemble(unittest.TestCase):
    def test_joins_in_id_order(self):
        shards = E.plan_shards([5, 6, 7], 2)
        a, b = unit(2), unit(1)
        ids, v = E.assemble_shards(shards, {"shard-00000": a, "shard-00001": b})
        self.assertEqual(ids, [5, 6, 7])
        np.testing.assert_array_equal(v, np.concatenate([a, b]))

    def test_refuses_a_missing_or_short_shard(self):
        shards = E.plan_shards([5, 6, 7], 2)
        with self.assertRaises(ValueError):
            E.assemble_shards(shards, {"shard-00000": unit(2)})
        with self.assertRaises(ValueError):
            E.assemble_shards(shards, {"shard-00000": unit(2), "shard-00001": unit(2)})


if __name__ == "__main__":
    unittest.main()
