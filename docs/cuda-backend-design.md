# CUDA backend v0

**Status:** experimental, implemented for the bounded subset below. This is
not a stable ABI, a general CUDA runtime, or a production-readiness claim.

## Selected v0 contract

- `emit_cuda(source, function_name)` emits a deterministic `.cu` translation
  unit for one named pure tensor function.
- The generated module contains a row-major `f32` tensor struct, one CUDA
  elementwise kernel, and an `extern "C"` host wrapper.
- The host wrapper takes tensor input pointers and `f32` scalar values, writes
  to a caller-provided output tensor, and returns a status code. The caller
  owns all host-side input/output storage and must ignore output on failure.
- The wrapper checks for a CUDA device, transfers input/output buffers,
  launches the kernel, synchronizes, copies the result to the host, and frees
  its temporary device allocations.
- If no device is available, the wrapper returns `AETHER_CUDA_NO_DEVICE`.
  There is no implicit CPU fallback. Null host pointers and CUDA runtime
  failures have distinct nonzero statuses.

## Supported language subset

- Static, non-empty, unitless `Tensor<f32, Shape>` return and tensor
  parameters of exactly the same shape.
- Unitless `f32` scalar parameters and scalar broadcasting.
- Pure expressions using unary `+`/`-` and elementwise `+`, `-`, and `*`.
- Straight-line single-use tensor temporaries that the conservative typed-IR
  fusion pass can inline, or a direct elementwise tensor return.

`f64`, reductions, division, conversions, indexing, user calls, autodiff,
effects, and local scalar temporaries are rejected. The backend does not
silently omit functions: a target outside this subset raises a diagnostic.

## Numerical and build contract

The kernel evaluates the original typed-IR expression tree and casts each
operation back to `float`. CUDA builds used for parity validation must disable
multiply-add contraction and flush-to-zero (`--fmad=false --ftz=false`).
Differential tests compare finite `f32` results in ULPs and currently require
at most one ULP for their selected cases; the test set is representative, not
an exhaustive guarantee for all values/devices.

Example compile command:

```text
nvcc --std=c++14 --fmad=false --ftz=false generated.cu -o generated
```

The host compiler and CUDA driver/runtime must be installed and usable. A
successful compiler invocation alone does not establish that a device is
available. CUDA execution is opt-in through the generated wrapper; no
device-selection policy, stream API, memory pooling, or asynchronous contract
is defined in v0.

## Validation and remaining work

Structural tests cover deterministic emission, explicit error statuses,
rejection outside the subset, and direct expressions. Host-runtime shims cover
missing devices and representative device-query, allocation, copy, and cleanup
failures. When `nvcc`, an MSVC x64 environment on Windows, and an accessible
device are available, native tests compare generated kernels with both the
interpreter and C99 at sizes 1, 255, 256, 257, 513, and 16x16, including the
grid-stride boundary. The selected finite `f32` results are required to match
within one ULP; this remains representative coverage, not an exhaustive
guarantee.

Broader shapes/operations, reductions, `f64`, autodiff, multi-kernel scheduling,
device selection, cross-platform CUDA CI, and a stable public ABI require
separate design and conformance work.
