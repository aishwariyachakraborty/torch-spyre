# Copyright 2025-2026 The Torch-Spyre Authors.
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

from .constants import DEVICE_NAME
from .patches import enable_spyre_context, patch_inductor_fusions
from . import config

import threading
from functools import wraps

import torch
from typing import Any

from .propagate_hints import spyre_hint, get_op_hints  # noqa: F401
from torch_spyre.profiler._ffdc import CATEGORY_COMPILE_FRONTEND, try_collect

_autoload_lock = threading.Lock()

# Set for the duration of a Spyre ``compile_fx`` call (the ``_wrapper`` Spyre
# branch below). Read by the invoke_subgraph decomposition monkey-patch
# (torch_spyre/_monkey_patch.py) to decide whether a nested_compile_region /
# invoke_subgraph HOP body that inherits its decompositions (config is None)
# should be re-traced with the Spyre decomp table. The HOP body is re-traced
# INSIDE the compile_fx call (AOTAutograd / proxy tracing), so this flag is
# active exactly when the extract runs. Thread-local because compiles can run
# concurrently and this must not leak the Spyre table into an unrelated
# CPU-only compile on another thread.
_compile_state = threading.local()


def in_spyre_compile() -> bool:
    """True while a Spyre ``compile_fx`` is on the stack for this thread."""
    return getattr(_compile_state, "in_spyre_compile", False)


def _spyre_inner_compile(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around ``compile_fx_inner`` that pins a picklable ``get_decomp_fn``.

    Background: passing ``decompositions=<dict>`` to ``compile_fx`` causes it
    to wrap the dict in a local ``def get_decomp_fn`` closure (compile_fx.py).
    That closure is unpicklable, so the FX graph cache silently bypasses
    itself with ``BypassFxGraphCache("Failed to pickle cache key")``.

    Two-stage decomposition design (these are not contradictory):

    * Outer stage — ``enable_spyre_compile_fx_wrapper``'s ``_wrapper`` DOES pass
      ``decompositions=get_spyre_decomp_table()`` to ``compile_fx``. That dict
      only feeds AOTAutograd's joint-graph decomposition; it is consumed before
      the FX graph cache key is built, so its unpicklable ``get_decomp_fn``
      closure is never part of the cache key.

    * Inner stage — this wrapper (installed as ``inner_compile``) never receives
      ``decompositions=``. Instead it clobbers ``get_decomp_fn`` at call time
      with the module-level ``get_spyre_decomp_table`` — a picklable,
      name-resolvable callable — so the post-AOT inner compile decomposes with
      the same table while keeping the FX graph cache key picklable.

    NOTE: We are working on improving this in upstream PyTorch
    """
    from torch._inductor.compile_fx import compile_fx_inner
    from torch_spyre._inductor.decompositions import get_spyre_decomp_table

    kwargs["get_decomp_fn"] = get_spyre_decomp_table
    return _lift_spyre_constants_and_compile(compile_fx_inner, *args, **kwargs)


def _lift_spyre_constants_and_compile(compile_fn, gm, example_inputs, **kwargs):
    """Turn frozen Spyre-tensor constants back into graph inputs, then compile.

    Freezing (SPYRE_FREEZING=1) folds parameters into ``get_attr`` constants
    before ``inner_compile`` -- which is what licenses constant folding and
    concat-linear. But every Spyre pass that assigns or plans device layouts
    (propagate_layouts, optimize_restickify, hbm_pool_planning, the LX planner,
    ...) enumerates graph inputs and op outputs, and none knows about Inductor
    constants. So keep freezing's rewrites, and hand the folded results to the
    rest of the pipeline as ordinary inputs: each Spyre-resident ``get_attr``
    becomes a trailing placeholder, and the compiled callable appends the
    constant tensors to its arguments at call time.

    Real inputs are extended in the same order: propagate_layouts pairs
    ``graph_input_names`` with ``V.get_real_inputs()`` positionally to read each
    input's device layout. With no Spyre constants (freezing off) this is a
    no-op passthrough.
    """
    lifted = [
        n
        for n in gm.graph.nodes
        if n.op == "get_attr"
        and isinstance(getattr(gm, n.target, None), torch.Tensor)
        and getattr(gm, n.target).device.type == DEVICE_NAME
    ]
    if not lifted:
        return compile_fn(gm, example_inputs, **kwargs)

    from torch._guards import detect_fake_mode
    from torch._inductor.virtualized import V

    consts = [getattr(gm, n.target) for n in lifted]
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    anchor = placeholders[-1] if placeholders else next(iter(gm.graph.nodes))
    for node in lifted:
        with (
            gm.graph.inserting_after(anchor)
            if placeholders
            else gm.graph.inserting_before(anchor)
        ):
            ph = gm.graph.placeholder(f"spyre_frozen_{node.target}")
        ph.meta.update(node.meta)
        node.replace_all_uses_with(ph)
        gm.graph.erase_node(node)
        anchor, placeholders = ph, [ph]
    gm.recompile()

    fake_mode = detect_fake_mode(example_inputs)
    # static_shapes: a constant's shape is fixed by definition. Left to the fake
    # mode's shape env, from_tensor may allocate symbolic sizes, and the symbols
    # then surface as non-integer loop ranges in the LX scratchpad planner.
    fake_consts = [
        fake_mode.from_tensor(c, static_shapes=True) if fake_mode else c for c in consts
    ]
    n_orig = len(example_inputs)
    new_inputs = list(example_inputs) + fake_consts
    # Constants never change between calls; let Inductor treat them as static.
    kwargs["static_input_idxs"] = list(kwargs.get("static_input_idxs") or []) + list(
        range(n_orig, n_orig + len(consts))
    )

    real = V.get_real_inputs()
    real_inputs = list(real[:n_orig]) if real else []
    with V.set_real_inputs(real_inputs + consts):
        compiled = compile_fn(gm, new_inputs, **kwargs)

    def _call_with_constants(args):
        args = list(args)
        args.extend(consts)
        return compiled(args)

    _call_with_constants._boxed_call = True  # type: ignore[attr-defined]
    return _call_with_constants


def enable_spyre_compile_fx_wrapper():
    import torch._inductor.compile_fx as cfx
    import torch.fx as fx
    import torch

    if getattr(cfx, "_spyre_wrapped", False):
        return
    with _autoload_lock:
        if getattr(cfx, "_spyre_wrapped", False):
            return

        patch_inductor_fusions()

        # Freezing has to be on before Dynamo traces, not just inside
        # compile_fx: Dynamo reads torch._inductor.config.freezing at trace time
        # to decide whether module parameters become real parameters (filling
        # TracingContext.params_flat) or plain graph inputs. Set late, in
        # enable_spyre_context's new_config, freeze() sees params_flat == 0 and
        # keeps every weight as a runtime input, freezing nothing. The cost is
        # that this is process-wide: with SPYRE_FREEZING=1 non-Spyre compiles
        # freeze too, which is why the flag is opt-in.
        if config.spyre_freezing:
            torch._inductor.config.freezing = True

        _orig = cfx.compile_fx
        from torch_spyre._inductor.logging_utils import get_inductor_logger

        logger = get_inductor_logger("compile_fx_wrapper")

        # Iterate over producer nodes (supports nested containers of nodes)
        def iter_nodes(x):
            if isinstance(x, fx.Node):
                yield x
            elif isinstance(x, (tuple, list)):
                for e in x:
                    yield from iter_nodes(e)
            elif isinstance(x, dict):
                for e in x.values():
                    yield from iter_nodes(e)

        def iter_tensors(v):
            if isinstance(v, torch.Tensor):
                yield v  # FakeTensor is a Tensor subclass, so this works
            elif isinstance(v, (tuple, list)):
                for e in v:
                    yield from iter_tensors(e)
            elif isinstance(v, dict):
                for e in v.values():
                    yield from iter_tensors(e)

        def _uses_spyre(gm, example_inputs, device_name=DEVICE_NAME) -> bool:
            # Inputs
            if any(
                isinstance(x, torch.Tensor)
                and getattr(x.device, "type", None) == device_name
                for x in (example_inputs or ())
            ):
                return True
            # Output
            out_node = gm.graph.output_node()
            out_puts = out_node.args[0] if out_node.args else []
            for n in iter_nodes(out_puts):
                meta = getattr(n, "meta", {}) or {}
                mv = meta.get("val", None) or meta.get("example_value", None)
                if mv is None:
                    continue

                if any(
                    getattr(getattr(t, "device", None), "type", None) == device_name
                    for t in iter_tensors(mv)
                ):
                    return True

            # Graph nodes (covers tensorless factories)
            for n in gm.graph.nodes:
                dev = n.kwargs.get("device")
                if dev is None:
                    continue

                if isinstance(dev, torch.device) and dev.type == device_name:
                    return True
                if isinstance(dev, str) and dev.split(":")[0] == device_name:
                    return True
            return False

        @wraps(_orig)
        def _wrapper(gm, example_inputs, *args, **kwargs):
            uses_spyre = _uses_spyre(gm, example_inputs)

            try:
                if uses_spyre:
                    torch.spyre._impl._lazy_init()

                    # AOTAutograd uses the dict passed via ``decompositions=``
                    # to decompose the joint graph; Spyre-specific
                    # decompositions must be applied at this stage so ops like
                    # aten.logical_not / aten.ceil / aten.sign are reduced to
                    # primitives the Spyre OpFuncs handler implements.
                    from torch_spyre._inductor.decompositions import (
                        get_spyre_decomp_table,
                    )

                    kwargs.setdefault("decompositions", get_spyre_decomp_table())
                    # Route inner compilation through _spyre_inner_compile,
                    # which re-binds ``get_decomp_fn`` to a picklable
                    # module-level callable so the FX graph cache key stays
                    # serializable.
                    kwargs.setdefault("inner_compile", _spyre_inner_compile)
                    # Mark this thread as inside a Spyre compile so the
                    # invoke_subgraph HOP re-trace (which happens within this
                    # _orig call) threads the Spyre decomp table into its
                    # subgraph bodies. Save/restore to stay correct under
                    # nested compile_fx calls.
                    _prev_in_spyre = getattr(_compile_state, "in_spyre_compile", False)
                    _compile_state.in_spyre_compile = True
                    try:
                        with enable_spyre_context(example_inputs):
                            return _orig(gm, example_inputs, *args, **kwargs)
                    finally:
                        _compile_state.in_spyre_compile = _prev_in_spyre

                # Non-Spyre graphs: no FFDC — avoids capturing unrelated CPU
                # compiles.
                return _orig(gm, example_inputs, *args, **kwargs)
            except Exception as exc:
                if uses_spyre:
                    try_collect(
                        exc, logger=logger, failure_category=CATEGORY_COMPILE_FRONTEND
                    )
                raise

        cfx.compile_fx = _wrapper
        cfx._spyre_wrapped = True


def _light_autoload():
    from . import decompositions  # noqa: F401
    from . import distributed as _distributed_init  # noqa: F401  registers spyre::broadcast_async/wait_work

    enable_spyre_compile_fx_wrapper()


def _autoload():
    if getattr(_autoload, "_ran", False):
        return

    with _autoload_lock:
        if getattr(_autoload, "_ran", False):
            return
        from torch._dynamo.device_interface import register_interface_for_device

        from torch_spyre.device.interface import SpyreInterface

        register_interface_for_device(DEVICE_NAME, SpyreInterface)

        from torch._inductor.codegen.common import (
            register_backend_for_device,
            register_device_op_overrides,
        )

        # Register in-tree CPU and CUDA device
        from torch._inductor.codegen import cpu_device_op_overrides  # noqa: F401  # usort: skip
        from torch._inductor.codegen.cuda import device_op_overrides  # noqa: F401  # usort: skip

        from torch_spyre.device.op_overrides import SpyreDeviceOpOverrides

        register_device_op_overrides(
            device=DEVICE_NAME, device_op_overrides=SpyreDeviceOpOverrides()
        )

        from .scheduler import SuperDSCScheduling
        from .wrapper import SpyrePythonWrapperCodegen

        register_backend_for_device(
            DEVICE_NAME,
            SuperDSCScheduling,
            SpyrePythonWrapperCodegen,
            device_custom_config=config,
        )

        _autoload._ran = True
