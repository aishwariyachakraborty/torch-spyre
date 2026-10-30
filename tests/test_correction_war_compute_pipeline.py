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

"""Ordering tests for program-correction and job-plan H2D uploads.

Every launch of a compiled kernel writes its program correction (which encodes
that launch's tensor addresses) to the same device region. If launch N+1's
correction upload overtakes launch N's still-running kernel, kernel N executes
with N+1's addresses: a write-after-read on the correction region (#3561).

The race can only happen while the host is queuing ahead of the device, so the
kernel here is a large matmul that runs much longer than one launch takes to
queue. Each burst checks, on the host clock, that queuing finished well before
the device did; if not, the test fails as "not stressed" instead of passing.

Run with and without SPYRE_H2D_COMPUTE_PIPELINE=1, which moves job-plan uploads
onto the COMPUTE pipeline. The flag is read once per process, so set it on the
pytest command line. SPYRE_WAR_TEST_SIZE overrides the matmul size (default
2048) if a machine needs a heavier kernel to build a queue.
"""

import math
import os
import time

import torch
import torch._dynamo
from torch.testing._internal.common_utils import TestCase

import torch_spyre  # noqa: F401

NUM_LAUNCHES = 16
SIZE = int(os.environ.get("SPYRE_WAR_TEST_SIZE", "2048"))
# Host must finish queuing all launches within this fraction of the time the
# device takes to finish them, i.e. the host ran several launches ahead.
MAX_ENQUEUE_FRACTION = 0.5


def _fn(x, y):
    return x @ y


def _make_inputs(n):
    # Scale y so outputs stay O(1) and fp16 error stays well inside tolerance.
    scale = 1.0 / math.sqrt(SIZE)
    pairs = []
    for i in range(n):
        g = torch.Generator().manual_seed(1000 + i)
        x = torch.randn(SIZE, SIZE, dtype=torch.float16, generator=g)
        y = (torch.randn(SIZE, SIZE, generator=g) * scale).to(torch.float16)
        pairs.append((x, y))
    return pairs


def _reference(pair):
    x, y = pair
    return x.float() @ y.float()


class TestCorrectionWarComputePipeline(TestCase):
    def setUp(self):
        torch._dynamo.reset()
        self.compiled = torch.compile(_fn, backend="inductor")
        self.mode = os.environ.get("SPYRE_H2D_COMPUTE_PIPELINE", "unset")

    def _burst(self, launch):
        """Run ``launch`` (which queues every launch), then synchronize.

        Returns the outputs and asserts the host was queuing ahead of the device.
        """
        t0 = time.perf_counter()
        outputs = launch()
        t_enqueue = time.perf_counter() - t0
        torch.spyre.synchronize()
        t_total = time.perf_counter() - t0
        frac = t_enqueue / t_total
        print(
            f"[war-stress] mode={self.mode} size={SIZE} launches={len(outputs)} "
            f"enqueue={t_enqueue * 1e3:.2f}ms total={t_total * 1e3:.2f}ms "
            f"enqueue/total={frac:.2f}"
        )
        self.assertLessEqual(
            frac,
            MAX_ENQUEUE_FRACTION,
            f"not stressed: host finished queuing at {frac:.0%} of the burst, so it "
            f"was not running ahead of the device. Raise SPYRE_WAR_TEST_SIZE "
            f"(now {SIZE}) instead of treating this run as evidence.",
        )
        return outputs

    def _assert_matches_cpu(self, outputs, pairs):
        refs = [_reference(p) for p in pairs]
        for i, out in enumerate(outputs):
            got = out.cpu().float()
            if torch.allclose(got, refs[i], atol=0.05, rtol=0.05):
                continue
            # Name the launch whose inputs the output actually came from: a
            # match with another launch is the signature of the race.
            errs = [(got - r).abs().max().item() for r in refs]
            closest = min(range(len(refs)), key=errs.__getitem__)
            self.fail(
                f"launch {i} of {len(pairs)} mismatched "
                f"(SPYRE_H2D_COMPUTE_PIPELINE={self.mode}): max abs err "
                f"{errs[i]:.3g} against its own inputs; closest match is launch "
                f"{closest} (err {errs[closest]:.3g})"
            )

    def test_queued_launches_with_distinct_corrections(self):
        """Each queued launch must use its own correction, not the next one's."""
        pairs = _make_inputs(NUM_LAUNCHES)
        device_pairs = [tuple(t.to("spyre") for t in p) for p in pairs]
        self.compiled(*device_pairs[0])  # compile outside the measured burst
        torch.spyre.synchronize()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            outputs = self._burst(lambda: [self.compiled(*p) for p in device_pairs])

        # Guard against a vacuous pass: the kernel must carry a host-computed
        # correction, or there is no correction upload to race.
        names = [event.name for event in prof.events()]
        self.assertGreaterEqual(names.count("launch::HostCompute"), NUM_LAUNCHES)
        self._assert_matches_cpu(outputs, pairs)

    def test_alternating_corrections_back_to_back(self):
        """A,B,A,B,... maximizes correction changes between adjacent launches."""
        pairs = _make_inputs(2)
        device_pairs = [tuple(t.to("spyre") for t in p) for p in pairs]
        self.compiled(*device_pairs[0])
        torch.spyre.synchronize()

        order = [i % 2 for i in range(NUM_LAUNCHES)]
        outputs = self._burst(lambda: [self.compiled(*device_pairs[i]) for i in order])
        self._assert_matches_cpu(outputs, [pairs[i] for i in order])

    def test_input_overwritten_between_queued_launches(self):
        """A user H2D into an input must not overtake the launch that reads it."""
        pairs = _make_inputs(NUM_LAUNCHES)
        x = pairs[0][0].to("spyre")
        y = pairs[0][1].to("spyre")
        self.compiled(x, y)
        torch.spyre.synchronize()

        # No queue-depth assertion here: a user copy_ may wait for the launch
        # that reads the same tensor, so this burst is ordered by design.
        outputs = []
        for cpu_x, cpu_y in pairs:
            x.copy_(cpu_x)
            y.copy_(cpu_y)
            outputs.append(self.compiled(x, y))
        torch.spyre.synchronize()
        self._assert_matches_cpu(outputs, pairs)
