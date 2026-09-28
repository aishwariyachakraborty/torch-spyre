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

"""What SPYRE_FREEZING actually does to the graph Spyre's passes receive.

Runs on the Spyre device: the Spyre passes and layouts under test only run when
``_uses_spyre(gm, example_inputs)`` is true (torch_spyre/_inductor/__init__.py).

Freezing is toggled by patching ``torch._inductor.config.freezing`` around
``torch.compile`` -- exactly what SPYRE_FREEZING=1 does process-wide at install
time, and early enough for Dynamo to see it.

The graph is captured by wrapping ``CustomPostPasses.__call__`` rather than by
patching ``post_grad_custom_post_pass``, since ``new_config`` installs its own
CustomPostPasses inside any outer patch. Ops are recorded both on entry (after
upstream freezing, before any Spyre post-grad pass) and on exit (after
decompose_addmm / mm_to_bmm / bmm_unflatten).

Frozen Spyre constants reach CustomPostPasses as trailing ``spyre_frozen_*``
placeholders, not ``get_attr`` nodes: _spyre_inner_compile lifts them back to
inputs so Spyre's layout and planning passes handle them like any other input.

Run:
    pytest tests/test_freezing_graph.py -v -s
"""

import pytest
import torch
import torch.nn as nn

import torch_spyre  # noqa: F401
from torch_spyre._inductor.passes import CustomPostPasses

DEVICE = "spyre"
DTYPE = torch.float16


def _spyre_available() -> bool:
    try:
        return bool(torch.spyre.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _spyre_available(), reason="needs a Spyre device")


class _ThreeParallelLinears(nn.Module):
    """Three Linears over one shared activation -- the q/k/v concat-linear shape."""

    def __init__(self, in_features: int = 64, out_features: int = 64) -> None:
        super().__init__()
        self.q = nn.Linear(in_features, out_features)
        self.k = nn.Linear(in_features, out_features)
        self.v = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.q(x) + self.k(x) + self.v(x)


def _make():
    torch.manual_seed(0)
    mod = _ThreeParallelLinears().to(DEVICE, DTYPE).eval()
    example = torch.randn(8, 64, dtype=DTYPE).to(DEVICE)
    return mod, example


def _summarize(graph):
    return {
        "ops": [str(n.target) for n in graph.nodes if n.op == "call_function"],
        "get_attr": sum(1 for n in graph.nodes if n.op == "get_attr"),
        "placeholders": sum(1 for n in graph.nodes if n.op == "placeholder"),
        "frozen": sum(
            1
            for n in graph.nodes
            if n.op == "placeholder" and str(n.target).startswith("spyre_frozen_")
        ),
    }


def _compile_and_capture(monkeypatch, mod, example, *, freezing: bool):
    """Compile fresh and return (output, before, after) for the first forward graph.

    ``before`` is the graph as CustomPostPasses receives it; ``after`` is what it
    hands on to lowering. The graph is mutated in place, so each is summarized
    at the moment it is taken rather than kept as a graph object.
    """
    captured = {}
    orig_call = CustomPostPasses.__call__

    def _recording_call(self, graph):
        first = "before" not in captured
        if first:
            captured["before"] = _summarize(graph)
        result = orig_call(self, graph)
        if first:
            captured["after"] = _summarize(graph)
        return result

    monkeypatch.setattr(CustomPostPasses, "__call__", _recording_call)

    # Without these, a second compile of the same forward reuses Dynamo's cached
    # frame (or an FX/AOT cache entry) and post-grad never runs again.
    torch._dynamo.reset()
    with torch._inductor.config.patch(freezing=freezing, force_disable_caches=True):
        compiled = torch.compile(mod, backend="inductor")
        with torch.no_grad():
            out = compiled(example)

    monkeypatch.setattr(CustomPostPasses, "__call__", orig_call)
    assert "before" in captured, (
        "CustomPostPasses never ran -- the compile did not take the Spyre path "
        "(check _uses_spyre) or did not reach post-grad."
    )
    print(f"\n[freezing={freezing}] before: {captured['before']}")
    print(f"[freezing={freezing}] after : {captured['after']}")
    return out, captured["before"], captured["after"]


def _gemms(summary) -> int:
    return sum(1 for op in summary["ops"] if "mm" in op)


def test_parameters_become_constants_under_freezing(monkeypatch):
    """Freezing folds the parameters, and the results arrive as lifted inputs.

    Unfrozen: 6 parameter placeholders + 1 activation, each weight transposed at
    runtime. Frozen: the transposes are folded away, and what remains of the
    weights arrives as ``spyre_frozen_*`` placeholders.
    """
    mod, example = _make()
    _, frozen, _ = _compile_and_capture(monkeypatch, mod, example, freezing=True)
    _, thawed, _ = _compile_and_capture(monkeypatch, mod, example, freezing=False)

    def permutes(summary):
        return sum(1 for op in summary["ops"] if "permute" in op)

    assert frozen["frozen"] > 0, frozen
    assert thawed["frozen"] == 0, thawed
    assert permutes(frozen) < permutes(thawed), (frozen["ops"], thawed["ops"])


def test_concat_linear_merges_parallel_linears(monkeypatch):
    """Three parallel Linears collapse into fewer GEMMs before Spyre's passes run.

    On a Spyre tensor check_concat_weights' CPU gate
    (``is_cpu and not config.cpp.enable_concat_linear``) short-circuits, so no
    cpp flag is needed.
    """
    mod, example = _make()
    _, frozen, _ = _compile_and_capture(monkeypatch, mod, example, freezing=True)
    _, thawed, _ = _compile_and_capture(monkeypatch, mod, example, freezing=False)

    assert _gemms(frozen) < _gemms(thawed), (frozen["ops"], thawed["ops"])


def test_no_addmm_reaches_lowering(monkeypatch):
    """No addmm may leave CustomPostPasses: lowering.py has no addmm rule.

    decompose_addmm runs first in CustomPostPasses. If this fails, concat-linear
    produced an addmm form decompose_addmm does not recognize, and that is the
    Part 2 fix.
    """
    mod, example = _make()
    _, _, after = _compile_and_capture(monkeypatch, mod, example, freezing=True)

    leftover = [op for op in after["ops"] if "addmm" in op]
    assert not leftover, after["ops"]


def test_numerics_frozen_matches_unfrozen(monkeypatch):
    """Freezing must not change results on device."""
    mod, example = _make()
    frozen_out, _, _ = _compile_and_capture(monkeypatch, mod, example, freezing=True)
    thawed_out, _, _ = _compile_and_capture(monkeypatch, mod, example, freezing=False)

    torch.testing.assert_close(frozen_out.cpu(), thawed_out.cpu(), rtol=1e-2, atol=1e-2)
