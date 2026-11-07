"""Compare native runtime of the conservative C99 tensor fusion pass."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

from src.api import compile_source
from src.c99 import emit_c99_module


_WORKLOADS = {
    "scale_shift_restore": (
        "let scaled = values * scale;",
        "let shifted = scaled + offset;",
        "let restored = shifted - values;",
        "return restored * scale;",
    ),
    "offset_cancel": (
        "let shifted = values + offset;",
        "let restored = shifted - offset;",
        "let scaled = restored * scale;",
        "return scaled + offset;",
    ),
    "negated_scale": (
        "let scaled = values * scale;",
        "let negated = -scaled;",
        "let shifted = negated + offset;",
        "return shifted - values;",
    ),
}


@dataclass(frozen=True)
class FusionBenchmarkResult:
    workload: str
    compiler: str
    elements: int
    loops_per_sample: int
    samples: int
    baseline_median_ms: float
    fused_median_ms: float
    speedup: float


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least one")


def _resolve_compiler(compiler: str | os.PathLike[str] | None) -> str:
    configured = compiler or os.environ.get("AETHER_C99_COMPILER")
    candidates = (os.fspath(configured),) if configured else ("gcc", "clang")
    for candidate in candidates:
        path = Path(candidate).expanduser()
        resolved = str(path.resolve()) if path.is_file() else shutil.which(candidate)
        if resolved is not None:
            return resolved
    raise RuntimeError(
        "No C compiler found; set AETHER_C99_COMPILER or pass --compiler"
    )


def _source(elements: int, workload: str) -> str:
    try:
        statements = _WORKLOADS[workload]
    except KeyError as error:
        raise ValueError(f"unknown fusion workload: {workload}") from error
    return (
        f"fn {workload}(values: Tensor<f32, [{elements}]>, "
        "scale: f32, offset: f32) -> Tensor<f32, "
        f"[{elements}]> {{\n"
        + "".join(f"  {statement}\n" for statement in statements)
        + "}\n"
    )


def _harness(elements: int, workload: str) -> str:
    function_symbol = "aether_fn_" + workload.encode("utf-8").hex()
    return f"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {{
    if (argc != 2) return 2;
    char *end = NULL;
    unsigned long long loops = strtoull(argv[1], &end, 10);
    if (end == argv[1] || *end != '\\0' || loops == 0) return 3;
    aether_tensor_f32_{elements} input = {{0}};
    for (size_t index = 0; index < {elements}; ++index)
        input.data[index] = (float)((int)(index % 31) - 15) / 8.0f;
    double checksum = 0.0;
    clock_t start = clock();
    for (unsigned long long iteration = 0; iteration < loops; ++iteration) {{
        size_t index = (size_t)(iteration % {elements});
        input.data[index] += 0.0001f;
        aether_tensor_f32_{elements} output =
            {function_symbol}(input, 1.0001f, 0.25f);
        checksum += (double)output.data[index];
    }}
    clock_t elapsed = clock() - start;
    printf("%.9f %.17g\\n", (double)elapsed / CLOCKS_PER_SEC, checksum);
    return 0;
}}
"""


def _compile(
    compiler: str,
    source: str,
    output: Path,
) -> None:
    source_path = output.with_suffix(".c")
    source_path.write_text(source, encoding="utf-8", newline="\n")
    command = [
        compiler,
        "-std=c99",
        "-pedantic",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O2",
        "-ffp-contract=off",
        str(source_path),
        "-o",
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(
            f"C99 benchmark compilation failed: {result.stderr or result.stdout}"
        )


def measure_fusion(
    *,
    compiler: str | os.PathLike[str] | None = None,
    workload: str = "scale_shift_restore",
    elements: int = 4096,
    loops_per_sample: int = 2000,
    samples: int = 7,
) -> FusionBenchmarkResult:
    """Measure fused and baseline native code; no timing threshold is assumed."""
    _validate_positive("elements", elements)
    _validate_positive("loops_per_sample", loops_per_sample)
    _validate_positive("samples", samples)
    if workload not in _WORKLOADS:
        raise ValueError(f"unknown fusion workload: {workload}")
    compiler_path = _resolve_compiler(compiler)
    module = compile_source(_source(elements, workload))
    baseline_c = emit_c99_module(module, optimize=False) + _harness(elements, workload)
    fused_c = emit_c99_module(module) + _harness(elements, workload)

    with tempfile.TemporaryDirectory(prefix="aether-fusion-bench-") as directory:
        root = Path(directory)
        baseline_exe = root / ("baseline.exe" if os.name == "nt" else "baseline")
        fused_exe = root / ("fused.exe" if os.name == "nt" else "fused")
        _compile(compiler_path, baseline_c, baseline_exe)
        _compile(compiler_path, fused_c, fused_exe)

        timings: dict[str, list[float]] = {"baseline": [], "fused": []}
        checksums: dict[str, list[float]] = {"baseline": [], "fused": []}
        executables = {"baseline": baseline_exe, "fused": fused_exe}
        for sample in range(samples):
            order = ("baseline", "fused") if sample % 2 == 0 else ("fused", "baseline")
            for name in order:
                start = time.perf_counter()
                result = subprocess.run(
                    [str(executables[name]), str(loops_per_sample)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                elapsed = time.perf_counter() - start
                if result.returncode:
                    raise RuntimeError(
                        f"{name} benchmark exited {result.returncode}: "
                        f"{result.stderr or result.stdout}"
                    )
                try:
                    measured_seconds, checksum = map(
                        float, result.stdout.split()
                    )
                except ValueError as error:
                    raise RuntimeError(
                        f"Unexpected {name} benchmark output: {result.stdout!r}"
                    ) from error
                timings[name].append(measured_seconds or elapsed)
                checksums[name].append(checksum)

    for baseline, fused in zip(checksums["baseline"], checksums["fused"]):
        if baseline != fused:
            raise RuntimeError(
                "Fused and baseline benchmark results differ "
                f"({fused!r} != {baseline!r})"
            )
    baseline_median = statistics.median(timings["baseline"])
    fused_median = statistics.median(timings["fused"])
    return FusionBenchmarkResult(
        workload=workload,
        compiler=compiler_path,
        elements=elements,
        loops_per_sample=loops_per_sample,
        samples=samples,
        baseline_median_ms=baseline_median * 1000.0,
        fused_median_ms=fused_median * 1000.0,
        speedup=(baseline_median / fused_median) if fused_median else 0.0,
    )


def measure_fusion_suite(
    *,
    compiler: str | os.PathLike[str] | None = None,
    elements: int = 4096,
    loops_per_sample: int = 2000,
    samples: int = 7,
) -> tuple[FusionBenchmarkResult, ...]:
    """Run all representative workloads and check fused/unfused checksums."""
    return tuple(
        measure_fusion(
            compiler=compiler,
            workload=workload,
            elements=elements,
            loops_per_sample=loops_per_sample,
            samples=samples,
        )
        for workload in _WORKLOADS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", help="C compiler executable")
    parser.add_argument(
        "--workload",
        choices=("all", *_WORKLOADS),
        default="all",
    )
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--loops", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if args.workload == "all":
        results = measure_fusion_suite(
            compiler=args.compiler,
            elements=args.elements,
            loops_per_sample=args.loops,
            samples=args.samples,
        )
    else:
        results = (
            measure_fusion(
                compiler=args.compiler,
                workload=args.workload,
                elements=args.elements,
                loops_per_sample=args.loops,
                samples=args.samples,
            ),
        )
    for result in results:
        print(f"Workload: {result.workload}")
        print(f"Compiler: {result.compiler}")
        print(
            f"Shape: {result.elements} f32 elements, "
            f"{result.loops_per_sample} calls/sample, {result.samples} samples"
        )
        print(f"Unfused median: {result.baseline_median_ms:.3f} ms")
        print(f"Fused median:   {result.fused_median_ms:.3f} ms")
        print(f"Observed speedup: {result.speedup:.3f}x")
        print("Fused and unfused checksums match.")
    print("Measurements are machine- and load-dependent, not a performance guarantee.")


if __name__ == "__main__":
    main()
