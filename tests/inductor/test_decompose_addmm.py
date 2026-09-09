# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for ``decompose_addmm`` (``torch_spyre._inductor.temp_passes``).

``decompose_addmm`` undoes Inductor's post-grad re-fusion of
``add(input, mm(a, b))`` back into ``aten.addmm.default``. There is no Spyre
lowering for ``addmm``, so a surviving one falls back to
``extern_kernels.addmm`` and yields an ``ExternKernelOut`` with no
``FixedTiledLayout``, which breaks later Spyre passes.

The pass operates on a raw ``torch.fx.Graph``, so these tests build graphs by
hand and call it directly -- no compile, no device.

The ``concat`` tests cover the shape freezing-time weight concatenation
produces: one wide ``addmm`` whose result is sliced back apart. Every slice
consumer has to keep its ``meta["val"]``, because the Spyre layout passes read
it; ``decompose_addmm`` rebuilds ``meta["val"]`` on each node it creates
(``temp_passes.py:346-382``), and these tests pin that.
"""

import unittest

import torch
from torch import nn

from torch_spyre._inductor.temp_passes import decompose_addmm


aten = torch.ops.aten


def _addmm_count(graph: torch.fx.Graph) -> int:
    return sum(
        1
        for n in graph.nodes
        if n.op == "call_function" and n.target is aten.addmm.default
    )


def _count(graph: torch.fx.Graph, target) -> int:
    return sum(
        1 for n in graph.nodes if n.op == "call_function" and n.target is target
    )


def _nodes(graph: torch.fx.Graph, target) -> list[torch.fx.Node]:
    return [
        n for n in graph.nodes if n.op == "call_function" and n.target is target
    ]


class _SingleAddmm(nn.Module):
    """``bias + x @ w`` in the re-fused ``addmm`` form."""

    def forward(
        self, bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        return aten.addmm.default(bias, x, w)


class _ConcatenatedAddmm(nn.Module):
    """The shape concat-linear leaves behind for Granite's GQA projections.

    One ``addmm`` of width 4096 + 1024 + 1024 = 6144, sliced back into q, k
    and v. Each slice is consumed so none is dead-code-eliminated.
    """

    Q_OUT = 4096
    KV_OUT = 1024
    TOTAL = Q_OUT + 2 * KV_OUT

    def forward(
        self, bias: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        wide = aten.addmm.default(bias, x, w)
        q = aten.slice.Tensor(wide, 1, 0, self.Q_OUT)
        k = aten.slice.Tensor(wide, 1, self.Q_OUT, self.Q_OUT + self.KV_OUT)
        v = aten.slice.Tensor(wide, 1, self.Q_OUT + self.KV_OUT, self.TOTAL)
        return (aten.relu.default(q), aten.relu.default(k), aten.relu.default(v))


def _traced_with_meta(
    mod: nn.Module, args: tuple[torch.Tensor, ...]
) -> torch.fx.GraphModule:
    """Trace ``mod`` and populate ``meta["val"]`` on every node.

    ``decompose_addmm`` reads ``node.meta["val"]`` to rebuild metadata on the
    nodes it creates, so a graph without it would exercise only the
    ``out_meta is None`` branch and silently skip what these tests check.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    with torch.no_grad():
        return make_fx(mod, tracing_mode="fake")(*args)


class TestDecomposeAddmm(unittest.TestCase):
    """The base case: one ``addmm`` becomes ``mm`` + ``add``."""

    def setUp(self) -> None:
        torch.manual_seed(0xBEEF)

    def test_single_addmm_decomposes(self) -> None:
        gm = _traced_with_meta(
            _SingleAddmm(),
            (torch.randn(32), torch.randn(8, 16), torch.randn(16, 32)),
        )
        self.assertEqual(_addmm_count(gm.graph), 1, "precondition: one addmm")

        decompose_addmm(gm.graph)

        self.assertEqual(
            _addmm_count(gm.graph), 0, "no addmm may survive the pass"
        )
        self.assertEqual(_count(gm.graph, aten.mm.default), 1)
        self.assertEqual(_count(gm.graph, aten.add.Tensor), 1)

    def test_decomposed_nodes_retain_meta_val(self) -> None:
        """Every node the pass creates carries a ``meta["val"]``."""
        gm = _traced_with_meta(
            _SingleAddmm(),
            (torch.randn(32), torch.randn(8, 16), torch.randn(16, 32)),
        )
        decompose_addmm(gm.graph)

        for target in (aten.mm.default, aten.add.Tensor):
            for node in _nodes(gm.graph, target):
                self.assertIn(
                    "val",
                    node.meta,
                    f"{target} node {node.name!r} lost meta['val']; the Spyre "
                    "layout passes read it",
                )


class TestDecomposeAddmmConcat(unittest.TestCase):
    """The concatenated form freezing-time weight concatenation produces."""

    def setUp(self) -> None:
        torch.manual_seed(0xBEEF)
        self.args = (
            torch.randn(_ConcatenatedAddmm.TOTAL),
            torch.randn(8, 4096),
            torch.randn(4096, _ConcatenatedAddmm.TOTAL),
        )

    def test_decompose_addmm_survives_concat(self) -> None:
        """A wide concatenated ``addmm`` with slice consumers decomposes cleanly.

        The pass must produce one ``mm`` + one ``add`` and leave the three
        slices intact, still reading the decomposed result.
        """
        gm = _traced_with_meta(_ConcatenatedAddmm(), self.args)
        self.assertEqual(_addmm_count(gm.graph), 1, "precondition: one wide addmm")
        slices_before = _count(gm.graph, aten.slice.Tensor)
        self.assertEqual(slices_before, 3, "precondition: three slices")

        decompose_addmm(gm.graph)  # calls graph.lint() internally

        self.assertEqual(_addmm_count(gm.graph), 0)
        self.assertEqual(_count(gm.graph, aten.mm.default), 1)
        self.assertEqual(_count(gm.graph, aten.add.Tensor), 1)
        self.assertEqual(
            _count(gm.graph, aten.slice.Tensor),
            slices_before,
            "the slices that read the concatenated result must survive",
        )

    def test_concat_slice_consumers_retain_meta_val(self) -> None:
        """Every consumer of the decomposed wide result keeps ``meta["val"]``.

        This is the assertion the plan singles out: concat-linear widens the
        GEMM, and if the decomposition dropped metadata on the path to the
        slices, the Spyre layout passes would fail downstream rather than here.
        """
        gm = _traced_with_meta(_ConcatenatedAddmm(), self.args)
        decompose_addmm(gm.graph)

        add_nodes = _nodes(gm.graph, aten.add.Tensor)
        self.assertEqual(len(add_nodes), 1)
        add_node = add_nodes[0]
        self.assertIn("val", add_node.meta)

        consumers = list(add_node.users)
        self.assertEqual(
            len(consumers), 3, f"expected the three slices, got {consumers}"
        )
        for consumer in consumers:
            self.assertIs(consumer.target, aten.slice.Tensor)
            self.assertIn(
                "val",
                consumer.meta,
                f"slice consumer {consumer.name!r} lost meta['val']",
            )

    def test_decomposed_mm_width_matches_concatenation(self) -> None:
        """The decomposed ``mm`` is the full concatenated width, not a slice of it.

        A regression that decomposed per-slice instead of once would still
        produce ``mm`` + ``add`` nodes and pass the count assertions, so pin the
        shape.
        """
        gm = _traced_with_meta(_ConcatenatedAddmm(), self.args)
        decompose_addmm(gm.graph)

        mm_nodes = _nodes(gm.graph, aten.mm.default)
        self.assertEqual(len(mm_nodes), 1)
        val = mm_nodes[0].meta.get("val")
        self.assertIsNotNone(val, "mm node lost meta['val']")
        self.assertEqual(
            val.shape[-1],
            _ConcatenatedAddmm.TOTAL,
            f"the mm must span the concatenated width "
            f"{_ConcatenatedAddmm.TOTAL}, got {tuple(val.shape)}",
        )


if __name__ == "__main__":
    unittest.main()
