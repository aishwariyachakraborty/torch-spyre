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

"""Device-free tests for the opt-in Inductor freezing flag (SPYRE_FREEZING).

These assert the *plumbing* only: that ``spyre_freezing`` reaches
``torch._inductor.config.freezing`` through ``enable_spyre_context``, that it is
off by default, and that the surrounding config is untouched either way. Whether
folding then actually fires, and what it does to the graph, needs a real compile
-- see ``test_freezing_graph.py``, which is skipped without a backend compiler.
"""

import torch

from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.patches import enable_spyre_context


def test_freezing_is_off_by_default():
    """The flag defaults off, so an unconfigured environment is unchanged."""
    assert spyre_config.spyre_freezing is False


def test_freezing_not_set_when_flag_off():
    """With the flag off, Spyre must not pin ``freezing`` at all.

    The assertion is that Spyre leaves Inductor's own default in place rather
    than writing False over it -- that is why patches.py spreads the key in
    conditionally instead of setting it to a boolean.
    """
    outer = torch._inductor.config.freezing
    with spyre_config.patch(spyre_freezing=False):
        with enable_spyre_context([]):
            assert torch._inductor.config.freezing == outer


def test_freezing_set_when_flag_on():
    """With the flag on, ``config.freezing`` is True inside the CM."""
    with spyre_config.patch(spyre_freezing=True):
        with enable_spyre_context([]):
            assert torch._inductor.config.freezing is True


def test_freezing_is_restored_on_exit():
    """``config.patch`` must restore the previous value when the CM exits."""
    before = torch._inductor.config.freezing
    with spyre_config.patch(spyre_freezing=True):
        with enable_spyre_context([]):
            pass
    assert torch._inductor.config.freezing == before


def test_parameters_are_not_discarded():
    """``freezing_discard_parameters`` stays off even with freezing enabled.

    Discarding parameters would leave the compiled module unable to reload its
    original state_dict. Enabling freezing must not opt into that silently.
    """
    with spyre_config.patch(spyre_freezing=True):
        with enable_spyre_context([]):
            assert torch._inductor.config.freezing_discard_parameters is False


def test_other_spyre_config_unchanged_by_freezing():
    """Enabling freezing must not perturb the rest of ``new_config``."""
    with spyre_config.patch(spyre_freezing=True):
        with enable_spyre_context([]):
            assert torch._inductor.config.split_reductions is False
            assert torch._inductor.config.permute_fusion is False
            assert torch._inductor.config.allow_buffer_reuse is False
            assert torch._inductor.config.fallback_random is True
            assert torch._inductor.config.unroll_reductions_threshold == 1
