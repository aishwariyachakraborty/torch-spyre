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

"""Device-free tests for the opt-in SPYRE_FREEZING flag.

The flag is read once, when the compile_fx wrapper is installed, and sets
``torch._inductor.config.freezing`` process-wide so Dynamo sees it at trace
time. Each case therefore runs in a fresh interpreter.
"""

import os
import subprocess
import sys

_PROBE = """
import torch
import torch_spyre
from torch_spyre._inductor import enable_spyre_compile_fx_wrapper
enable_spyre_compile_fx_wrapper()
c = torch._inductor.config
print(c.freezing, c.freezing_discard_parameters)
"""


def _probe(env_value):
    env = dict(os.environ)
    env.pop("SPYRE_FREEZING", None)
    if env_value is not None:
        env["SPYRE_FREEZING"] = env_value
    out = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True
    )
    freezing, discard = out.stdout.strip().splitlines()[-1].split()
    return freezing == "True", discard == "True"


def test_freezing_off_when_unset():
    assert _probe(None) == (False, False)


def test_freezing_off_when_zero():
    assert _probe("0") == (False, False)


def test_freezing_on_when_set():
    """On, and parameters are not discarded (state_dict stays reloadable)."""
    assert _probe("1") == (True, False)
