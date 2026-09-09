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

"""Tests for the Spyre freezing opt-in (``spyre_freezing`` / ``SPYRE_FREEZING``).

Freezing treats parameters as constants. That licenses two upstream behaviours
that Spyre wants and previously could not reach:

  - compile-time constant folding of parameter-only arithmetic, and
  - freezing-time weight concatenation, which fuses the linears that share one
    activation (a decoder layer's q/k/v projections, and its gate/up
    projections) into one wider GEMM each.

The upstream gate is device-independent -- ``compile_fx.py`` checks only
``config.freezing and not torch.is_grad_enabled()`` -- and it runs before
``inner_compile``, so there is no Spyre-side pass to add. The frozen graph
simply arrives at the Spyre post-grad hooks already folded. These tests
therefore split into two groups:

  - ``TestSpyreFreezingConfig`` -- the opt-in wiring itself: default off, on
    under ``SPYRE_FREEZING=1``, and never overriding a caller who set
    ``torch._inductor.config.freezing`` by hand. Needs no device and no
    compile.
  - ``TestConcatLinearOnFrozenWeights`` -- that concat-linear actually fires on
    Granite 3.3 8B's GQA projection shapes, asserted on the FX graph after
    freezing. Needs a compile but no device.

On the GQA shapes specifically: ``check_concat_weights``
(``torch/_inductor/fx_passes/freezing_patterns.py``) requires the candidate
weights to agree on ``meta["val"].shape[:-1]``. The pattern's weight operand is
the ``mm`` right-hand side, i.e. ``(in, out)`` -- the replacement concatenates
on ``dim=1`` and splits on ``w.size(1)`` -- so ``shape[:-1]`` is ``(in,)``,
which is 4096 for all of q, k and v. The dimension that differs across GQA
projections (out: 4096 / 1024 / 1024) is precisely the one the check excludes,
so all three fuse. ``nn.Linear``'s ``permute(weight)`` reaches the check as a
single ``get_attr`` constant because ``freezing_passes`` runs ``constant_fold``
before applying ``pass_patterns``.
"""

import os
import unittest
from unittest.mock import patch

import torch

from torch._inductor import config as t_inductor_config

from torch_spyre._inductor import config as ts_inductor_config
from torch_spyre._inductor.patches import enable_spyre_context


# ---------------------------------------------------------------------------
# Modules under test -- real Granite 3.3 8B decoder-layer projection shapes.
# ---------------------------------------------------------------------------


class QKVLike(torch.nn.Module):
    """Granite 3.3 8B attention projections: hidden 4096, 32 heads, 8 KV heads.

    GQA makes the three projections unequal in width -- q 4096, k 1024,
    v 1024 -- while all three read the same ``hidden_states``. That shared
    activation is what concat-linear keys on, and the unequal widths are what
    the module exists to exercise.
    """

    HIDDEN = 4096
    Q_OUT = 4096
    KV_OUT = 1024

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(self.HIDDEN, self.Q_OUT, bias=False)
        self.k_proj = torch.nn.Linear(self.HIDDEN, self.KV_OUT, bias=False)
        self.v_proj = torch.nn.Linear(self.HIDDEN, self.KV_OUT, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )


class GateUpLike(torch.nn.Module):
    """Granite 3.3 8B MLP projections: hidden 4096 -> 12800, twice (SwiGLU).

    Equal widths, so this is the pair concat-linear handles via the two-way
    pattern (``matmul_fuse_pattern_two``) rather than the three-way one.
    """

    HIDDEN = 4096
    INTERMEDIATE = 12800

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(self.HIDDEN, self.INTERMEDIATE, bias=False)
        self.up_proj = torch.nn.Linear(self.HIDDEN, self.INTERMEDIATE, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.gate_proj(hidden_states), self.up_proj(hidden_states))


class QKVWithBias(torch.nn.Module):
    """Same shapes as :class:`QKVLike` but with biases, so the fused node is an
    ``addmm`` rather than an ``mm``.

    This is the form ``decompose_addmm`` has to survive: concat-linear widens
    the GEMM and then slices it, and ``decompose_addmm`` splits the surviving
    ``addmm`` back into ``mm`` + ``add`` for the Spyre lowerings.
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(QKVLike.HIDDEN, QKVLike.Q_OUT, bias=True)
        self.k_proj = torch.nn.Linear(QKVLike.HIDDEN, QKVLike.KV_OUT, bias=True)
        self.v_proj = torch.nn.Linear(QKVLike.HIDDEN, QKVLike.KV_OUT, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )


# ---------------------------------------------------------------------------
# Config wiring -- no compile, no device.
# ---------------------------------------------------------------------------


class TestSpyreFreezingConfig(unittest.TestCase):
    """The opt-in itself: default, enable, and non-override."""

    def test_freezing_config_off_by_default(self) -> None:
        """``spyre_freezing`` is False without ``SPYRE_FREEZING``, and the
        Spyre config patch then does not mention ``freezing`` at all."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPYRE_FREEZING", None)
            # Re-read the env the same way config.py does at import time,
            # rather than trusting the already-imported module attribute.
            self.assertFalse(
                os.environ.get("SPYRE_FREEZING", "0") == "1",
                "SPYRE_FREEZING leaked into the test environment",
            )

        with ts_inductor_config.patch("spyre_freezing", False):
            self.assertFalse(ts_inductor_config.spyre_freezing)
            self.assertNotIn(
                "freezing",
                self._spyre_config_keys(),
                "freezing must be absent (not False) from the Spyre config patch "
                "when the opt-in is off",
            )

    def test_freezing_config_enables_inductor_freezing(self) -> None:
        """With the opt-in on, ``torch._inductor.config.freezing`` is True
        inside ``enable_spyre_context``."""
        with ts_inductor_config.patch("spyre_freezing", True):
            self.assertFalse(
                t_inductor_config.freezing,
                "precondition: upstream freezing is off outside the context",
            )
            with enable_spyre_context([]):
                self.assertTrue(
                    t_inductor_config.freezing,
                    "SPYRE_FREEZING=1 must turn on upstream Inductor freezing",
                )
            self.assertFalse(
                t_inductor_config.freezing,
                "the context manager must restore the previous value on exit",
            )

    def test_freezing_does_not_override_user_setting(self) -> None:
        """A caller who set ``torch._inductor.config.freezing`` themselves keeps
        it, even with the Spyre opt-in off.

        This is what the ``**({...} if ... else {})`` spread buys: the key is
        absent rather than False when the opt-in is off, and
        ``config.patch(dict)`` leaves absent keys untouched.
        """
        with ts_inductor_config.patch("spyre_freezing", False):
            with t_inductor_config.patch("freezing", True):
                with enable_spyre_context([]):
                    self.assertTrue(
                        t_inductor_config.freezing,
                        "the Spyre patch must not silently clear a caller's "
                        "own freezing=True",
                    )

    @staticmethod
    def _spyre_config_keys() -> set[str]:
        """The key set of the dict ``enable_spyre_context`` patches into
        ``torch._inductor.config``.

        Captured by intercepting ``torch._inductor.config.patch`` rather than
        duplicating the dict here, so this cannot drift from ``patches.py``.
        """
        captured: dict[str, object] = {}
        real_patch = t_inductor_config.patch

        def spy(*args, **kwargs):
            if args and isinstance(args[0], dict):
                captured.update(args[0])
            return real_patch(*args, **kwargs)

        with patch.object(t_inductor_config, "patch", spy):
            with enable_spyre_context([]):
                pass
        return set(captured)


# ---------------------------------------------------------------------------
# Concat-linear on frozen weights -- compiles, but needs no device.
# ---------------------------------------------------------------------------


class TestConcatLinearOnFrozenWeights(unittest.TestCase):
    """That freezing-time weight concatenation actually fires, on real shapes.

    These run the freezing FX passes directly on an exported graph rather than
    going through a full Spyre compile: the behaviour under test is upstream and
    device-independent, and asserting on the FX graph keeps the test honest
    about what it verified.
    """

    def setUp(self) -> None:
        torch.manual_seed(0xBEEF)
        self.patchers = [
            t_inductor_config.patch("force_disable_caches", True),
        ]
        for p in self.patchers:
            p.__enter__()
        torch.compiler.reset()

    def tearDown(self) -> None:
        for p in self.patchers:
            p.__exit__(None, None, None)
        torch.compiler.reset()

    @staticmethod
    def _frozen_graph(mod: torch.nn.Module, example: torch.Tensor):
        """Export ``mod`` to an FX graph and run upstream's freezing passes on it.

        Mirrors what ``fw_compiler_freezing`` does before ``inner_compile``:
        parameters become constants, then ``freezing_passes`` constant-folds and
        applies the concat-linear ``pass_patterns``.
        """
        from torch._inductor.fx_passes.freezing_patterns import freezing_passes

        with torch.no_grad():
            gm = torch.fx.symbolic_trace(mod)
            # symbolic_trace leaves parameters as get_attr already, which is the
            # state freezing produces; run the passes over it directly.
            freezing_passes(gm, [example])
        return gm

    @staticmethod
    def _count(gm, target) -> int:
        return sum(
            1
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is target
        )

    def test_concat_linear_fires_on_two_linears(self) -> None:
        """Two equal-width linears sharing an activation become one GEMM.

        ``GateUpLike`` is Granite's SwiGLU pair: 4096 -> 12800 twice. After
        freezing there must be one matmul, not two.
        """
        mod = GateUpLike().eval()
        x = torch.randn(8, GateUpLike.HIDDEN)
        gm = self._frozen_graph(mod, x)

        mms = self._count(gm, torch.ops.aten.mm.default)
        self.assertEqual(
            mms,
            1,
            f"expected the gate/up pair to fuse into 1 mm, found {mms}. "
            "Graph:\n" + gm.graph.python_code("self").src,
        )

    def test_concat_linear_fires_on_gqa_qkv(self) -> None:
        """All three GQA projections fuse, despite unequal output widths.

        ``check_concat_weights`` compares ``shape[:-1]`` of the ``mm`` weight
        operand, which is ``(in,)`` = (4096,) for q, k and v alike. The differing
        output width is the excluded dimension. If this ever regresses to a
        partial fusion (2 mms) that is a real finding about the upstream check,
        not a bug in this test -- the assertion message says so.
        """
        mod = QKVLike().eval()
        x = torch.randn(8, QKVLike.HIDDEN)
        gm = self._frozen_graph(mod, x)

        mms = self._count(gm, torch.ops.aten.mm.default)
        self.assertEqual(
            mms,
            1,
            f"expected q/k/v to fuse into 1 mm, found {mms}. GQA widths differ "
            "(4096/1024/1024) but check_concat_weights compares shape[:-1] of "
            "the (in, out) weight, which is (4096,) for all three. A count of 2 "
            "or 3 means that reading is wrong -- record it as a finding.\n"
            + gm.graph.python_code("self").src,
        )

    def test_folded_weights_are_get_attr_constants(self) -> None:
        """``permute(weight)`` folds to a single ``get_attr``.

        This is the precondition concat-linear's ``inp.op == "get_attr"`` test
        depends on, and the reason ``constant_fold`` runs before
        ``pass_patterns``. Asserting it separately means a regression here is
        distinguishable from a regression in the pattern itself.
        """
        mod = GateUpLike().eval()
        x = torch.randn(8, GateUpLike.HIDDEN)
        gm = self._frozen_graph(mod, x)

        permutes = self._count(gm, torch.ops.aten.permute.default)
        self.assertEqual(
            permutes,
            0,
            f"expected every weight permute to be constant-folded, found "
            f"{permutes}",
        )
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is torch.ops.aten.mm.default:
                weight = node.args[1]
                self.assertEqual(
                    weight.op,
                    "get_attr",
                    "the mm weight operand must be a folded get_attr constant, "
                    f"got op={weight.op!r} target={weight.target!r}",
                )


if __name__ == "__main__":
    unittest.main()
