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

"""What freezing actually does to the graph Spyre's passes receive.

The config test asserts the flag is wired. This asserts the consequence, which
is the part that decides whether freezing is useful on Spyre:

  1. Parameters arrive at the post-grad passes as ``get_attr`` constants rather
     than placeholders -- the precondition for every folding rewrite.
  2. Three ``nn.Linear`` layers sharing one activation are concatenated into one
     wider GEMM (concat-linear), and the weight count drops.
  3. The resulting op mix is one Spyre can lower. This is the known-risk test:
     concat-linear emits ``addmm``, for which ``lowering.py`` registers no
     lowering, so ``decompose_addmm`` must split it back into ``mm + add``.

Each test captures the graph via a post-grad hook rather than compiling all the
way down, so (1) and (2) need no device. Test (3) is the same hook, asserting on
the op set the Spyre lowerings will be asked for.

Run:
    pytest tests/test_freezing_graph.py -v
    SPYRE_FREEZING=1 pytest tests/test_freezing_graph.py -v   # same result; the
        tests set the flag themselves, so the env var is not required.
"""

import torch
import torch.nn as nn

from torch_spyre._inductor import config as spyre_config


class _ThreeParallelLinears(nn.Module):
    """Three Linears over one shared activation -- the concat-linear shape.

    This is the q/k/v projection shape: the pattern freezing's concat-linear is
    built to find, and the reason #32 was ranked worth doing.
    """

    def __init__(self, in_features: int = 64, out_features: int = 64) -> None:
        super().__init__()
        self.q = nn.Linear(in_features, out_features)
        self.k = nn.Linear(in_features, out_features)
        self.v = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.q(x) + self.k(x) + self.v(x)


def _capture_post_grad_graph(mod, example, *, freezing: bool):
    """Compile ``mod`` and return the post-grad FX graph Spyre's passes see.

    Hooks ``post_grad_custom_post_pass``, which is where Spyre installs
    CustomPostPasses -- so what this captures is exactly what Spyre receives,
    after upstream freezing has had its turn.
    """
    captured = {}

    def _capture(graph):
        # Record the first (forward) graph only; a backward graph would overwrite
        # the thing under test.
        captured.setdefault("graph", graph)
        return graph

    with spyre_config.patch(spyre_freezing=freezing):
        with torch._inductor.config.patch(post_grad_custom_post_pass=_capture):
            compiled = torch.compile(mod, backend="inductor")
            with torch.no_grad():
                compiled(example)

    assert "graph" in captured, (
        "post_grad_custom_post_pass never fired -- the compile did not reach "
        "post-grad, so nothing was captured. Check that torch.compile actually "
        "compiled (not eager-fallback) before reading into this failure."
    )
    return captured["graph"]


def _op_names(graph) -> list[str]:
    return [str(n.target) for n in graph.nodes if n.op == "call_function"]


def _count(graph, op_substring: str) -> int:
    return sum(1 for name in _op_names(graph) if op_substring in name)


def test_parameters_become_constants_under_freezing():
    """Freezing must turn parameters into get_attr constants.

    This is the precondition for all folding: with parameters still arriving as
    placeholders there is nothing for a folding pass to fold.
    """
    mod = _ThreeParallelLinears().eval()
    example = torch.randn(8, 64)

    frozen = _capture_post_grad_graph(mod, example, freezing=True)
    thawed = _capture_post_grad_graph(mod, example, freezing=False)

    frozen_attrs = sum(1 for n in frozen.nodes if n.op == "get_attr")
    thawed_attrs = sum(1 for n in thawed.nodes if n.op == "get_attr")

    assert frozen_attrs > thawed_attrs, (
        f"freezing produced no new get_attr constants "
        f"(frozen={frozen_attrs}, thawed={thawed_attrs}). Either freezing did "
        f"not run, or it ran and folded nothing."
    )


def test_concat_linear_merges_parallel_linears():
    """Three parallel Linears should collapse into fewer, wider GEMMs.

    NOTE on the gate, because it is easy to misread. check_concat_weights
    (torch/_inductor/fx_passes/freezing_patterns.py:127-129) reads
    ``match.kwargs["inp"].meta["val"].is_cpu`` -- the device of the MATCHED
    activation -- and only then requires ``config.cpp.enable_concat_linear``:

        is_cpu = match.kwargs["inp"].meta["val"].is_cpu
        if is_cpu and not config.cpp.enable_concat_linear:
            return False

    So on a CPU tensor this is off by default, and on a Spyre tensor the gate
    short-circuits and concat-linear runs. This test therefore runs on CPU and
    MUST enable the cpp flag to see the rewrite at all -- which is what makes it
    a test of the rewrite rather than of the gate. On device the flag is moot.
    """
    mod = _ThreeParallelLinears().eval()
    example = torch.randn(8, 64)

    # Required only because this test runs on CPU; see the docstring.
    with torch._inductor.config.patch({"cpp.enable_concat_linear": True}):
        frozen = _capture_post_grad_graph(mod, example, freezing=True)
        thawed = _capture_post_grad_graph(mod, example, freezing=False)

    def gemms(graph):
        return _count(graph, "mm") + _count(graph, "addmm")

    assert gemms(frozen) < gemms(thawed), (
        f"no GEMM merging under freezing (frozen={gemms(frozen)}, "
        f"thawed={gemms(thawed)}). Op names frozen: {_op_names(frozen)}"
    )


def test_no_addmm_survives_for_spyre_to_lower():
    """The known risk: concat-linear emits addmm, which Spyre cannot lower.

    ``lowering.py`` registers ``aten.mm.default`` and ``aten.bmm.default`` but
    not ``aten.addmm``; ``decompose_addmm`` in CustomPostPasses is what splits
    addmm into mm + add. This asserts that by the time the graph is handed on,
    no bare addmm is left.

    A failure here is the expected first failure of this work, and it is
    informative rather than fatal: it means concat-linear fires (good) but its
    output reaches lowering in a form Spyre has no rule for, and the fix is
    pass ordering -- decompose_addmm must run after folding, not before.
    """
    mod = _ThreeParallelLinears().eval()
    example = torch.randn(8, 64)

    with torch._inductor.config.patch({"cpp.enable_concat_linear": True}):
        frozen = _capture_post_grad_graph(mod, example, freezing=True)
    leftover = [n for n in _op_names(frozen) if "addmm" in n]

    assert not leftover, (
        f"addmm survived to the Spyre passes: {leftover}. lowering.py has no "
        f"addmm rule, so this must be decomposed first. See the docstring -- "
        f"this is a pass-ordering fix, not a premise failure."
    )


def test_numerics_match_eager_under_freezing():
    """Folding must not change results. Runs on CPU; no device needed."""
    mod = _ThreeParallelLinears().eval()
    example = torch.randn(8, 64)

    with torch.no_grad():
        expected = mod(example)

    with spyre_config.patch(spyre_freezing=True):
        compiled = torch.compile(mod, backend="inductor")
        with torch.no_grad():
            actual = compiled(example)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
