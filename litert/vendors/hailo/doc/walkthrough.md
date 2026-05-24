# Hailo In LiteRT: Implementation Summary and Build/Run Guide

This document summarizes the Hailo vendor integration in LiteRT and provides a practical workflow for building, packaging, and running Hailo-accelerated models.

## 1) Implementation Summary

### 1.1 Workspace/SDK wiring
- `third_party/hailo/hailo.bzl`
  - Declares the `hailo_sdk` repository using `HAILO_RT_DIR`.
- `third_party/hailo/hailo.bazel`
  - Exposes `@hailo_sdk//:hailort` from SDK headers and `libhailort.so`.
- `WORKSPACE`
  - Loads `hailo_configure()` so Bazel can use a real SDK instead of the mock SDK.

### 1.2 Hailo compiler plugin (AOT wrapping)
- `litert/vendors/hailo/compiler/hailo_compiler_plugin.cc`
  - Selects ops for one partition per subgraph.
  - Reads a precompiled HEF path from `LITERT_HAILO_HEF_PATH`.
  - Wraps HEF bytes into LiteRT compiled result bytecode modules.
  - Does not run Hailo DFC in-process.

### 1.3 Hailo dispatch runtime
- `litert/vendors/hailo/dispatch/dispatch_api.cc`
  - Exposes dispatch API entrypoints for LiteRT.
- `litert/vendors/hailo/dispatch/device_context.*`
  - Owns `hailort::VDevice` and tensor-buffer registration.
- `litert/vendors/hailo/dispatch/invocation_context.*`
  - Loads embedded HEF from dispatch bytecode.
  - Uses HailoRT InferModel flow for runtime execution.
  - Binds LiteRT host buffers to Hailo bindings and invokes inference.

### 1.4 Build stamp metadata
- LiteRT tags compiled models with `LiteRtStamp` metadata (soc manufacturer/model).
- This is expected and used by LiteRT to identify compiled models.

## 2) Build Prerequisites

- HailoRT SDK installed on target/build host.
- `HAILO_RT_DIR` points to an SDK root containing:
  - `include/hailo/hailort.hpp`
  - `lib/libhailort.so` (or equivalent platform path)

Example:

```bash
export HAILO_RT_DIR=/usr
```

On systems with multiple libhailort installs, ensure a single consistent version is first in runtime linker search path.

## 3) Build Hailo Plugin and Dispatch Libraries

```bash
cd /path/to/LiteRT
export HAILO_RT_DIR=/usr

bazel build //litert/vendors/hailo/compiler:compiler_plugin
bazel build //litert/vendors/hailo/dispatch:dispatch_api_so
```

Optional tests:

```bash
bazel test //litert/vendors/hailo/compiler:hailo_compiler_plugin_test
```

## 4) Prepare HEF Offline

Compile your base model to HEF using Hailo Dataflow Compiler on a supported host.

Important:
- Use an HEF that matches the same model interface (input/output contract) you will wrap.
- Validate HEF contract before wrapping:

```bash
hailortcli parse-hef /path/to/model.hef
```

## 5) Wrap HEF Into a LiteRT Model

Use `apply_plugin_main` in apply mode.

```bash
cd /path/to/LiteRT
export HAILO_RT_DIR=/usr

bazel run //litert/tools:apply_plugin_main -- \
  --cmd=apply \
  --model=/path/to/input_model.tflite \
  --soc_manufacturer=Hailo \
  --soc_model=Hailo-10H \
  --hailo_hef_path=/path/to/model.hef \
  --libs=bazel-bin/litert/vendors/hailo/compiler \
  --o=/path/to/output_model.hef.tflite
```

Notes:
- `--hailo_hef_path` is forwarded to `LITERT_HAILO_HEF_PATH`.
- The output model contains DISPATCH_OP plus embedded HEF bytecode.

## 6) Build and Install LiteRT Wheel (with Hailo artifacts)

```bash
cd /path/to/LiteRT
export HAILO_RT_DIR=/usr

bazel build //ci/tools/python/wheel:litert_wheel
pip uninstall -y ai-edge-litert
pip install bazel-bin/ci/tools/python/wheel/dist/ai_edge_litert-*.whl
```

Verify Hailo dispatch library is present in the wheel installation:

```bash
python - <<'PY'
import os, ai_edge_litert
p = os.path.dirname(ai_edge_litert.__file__)
print(os.path.join(p, 'vendors', 'hailo', 'dispatch'))
print(os.listdir(os.path.join(p, 'vendors', 'hailo', 'dispatch')))
PY
```

## 7) Run Inference

- Use the wrapped `.hef.tflite` model with LiteRT runtime APIs.
- Check logs for successful Hailo dispatch initialization:
  - `Loading ... libLiteRtDispatch_Hailo.so`
  - `Initializing Hailo NPU Dispatch API runtime`
  - `Hailo InvocationContext initialized successfully`

## 8) Common Issues and Fixes

### 8.1 Contract mismatch (most common)
- Symptom: invocation fails with input/output size mismatch.
- Cause: HEF I/O contract does not match wrapped model contract.
- Fix: regenerate HEF for the exact same model partition and re-wrap.

### 8.2 Device not found
- Verify with:
  - `hailortcli scan`
  - `ls -la /dev/hailo*`
  - `lsmod | grep hailo`

### 8.3 Multiple HailoRT versions installed
- Conflicting binaries/libraries can cause subtle failures.
- Ensure `hailortcli` and `libhailort.so` resolve to the same intended version.

## 9) Recommended Validation Flow

1. Validate HEF:
   - `hailortcli parse-hef model.hef`
2. Validate wrapped model build:
   - `apply_plugin_main --cmd=apply ...`
3. Validate runtime artifacts in wheel:
   - `libLiteRtDispatch_Hailo.so` present in site-packages
4. Run one-shot inference and then throughput runs.

