import os
import random
import sys
import types
import unittest


def _ensure_numpy_stub():
    """Provide a minimal numpy stub so Sumtree imports without the dependency."""

    if "numpy" in sys.modules:
        return

    numpy = types.ModuleType("numpy")
    numpy.abs = abs
    numpy.random = types.SimpleNamespace(
        randint=lambda low, high=None: random.randint(low, (high - 1) if high is not None else low)
    )
    sys.modules["numpy"] = numpy
    sys.modules["numpy.random"] = numpy.random


_ensure_numpy_stub()

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from Sumtree import STNode, SumTree  # noqa: E402
from Sumtree import PriorityQueue  # noqa: E402


def build_sample_tree():
    left_leaf = STNode.createLeaf(val=3, payload="left")
    mid_leaf = STNode.createLeaf(val=4, payload="middle")
    right_leaf = STNode.createLeaf(val=5, payload="right")

    left_internal = STNode(left_leaf, mid_leaf, val=left_leaf.val + mid_leaf.val)
    root = STNode(left_internal, right_leaf, val=left_internal.val + right_leaf.val)

    return SumTree(root)


class SumTreeRetrieveTests(unittest.TestCase):
    def setUp(self):
        self.tree = build_sample_tree()

    def test_value_exceeding_subtree_sum_raises_error(self):
        with self.assertRaises(ValueError):
            self.tree.retrieve(8, cur=self.tree.root.left_child)

    def test_retrieve_returns_correct_leaf_from_subtree(self):
        node = self.tree.retrieve(6, cur=self.tree.root.left_child)
        self.assertTrue(node.is_leaf)
        self.assertEqual(node.payload, "middle")

    def test_retrieve_returns_correct_leaf_from_root(self):
        node = self.tree.retrieve(9)
        self.assertTrue(node.is_leaf)
        self.assertEqual(node.payload, "right")


class PriorityQueueReplacementTests(unittest.TestCase):
    def test_replacement_respects_max_size_and_updates_leaf(self):
        queue = PriorityQueue(max_size=1)

        initial_leaf = STNode.createLeaf(val=1, payload="initial")
        queue.insert(new=initial_leaf)
        self.assertEqual(queue.size, 1)

        replacement_leaf = STNode.createLeaf(val=5, payload="replacement")
        queue.insert(new=replacement_leaf)

        self.assertEqual(queue.size, 1)
        self.assertEqual(queue.root.payload, "replacement")
        self.assertEqual(queue.root.val, 5)

        retrieved = queue.retrieve(queue.root.val)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.payload, "replacement")


if __name__ == "__main__":
    unittest.main()
