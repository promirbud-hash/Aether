"""Experimental CUDA C++ emission for the bounded f32 tensor subset."""

from __future__ import annotations

from dataclasses import dataclass
import sys

from src.api import compile_source
from src.frontend import Diagnostic
from src.fusion import fuse_elementwise_tensors
from src.ir import IRExpression, IRFunction, IRModule
from src.semantic import AetherType


_CUDA_BLOCK_SIZE = 256
_MAX_CUDA_ELEMENTS = sys.maxsize // 4


class CudaLoweringError(ValueError):
    """Raised when a function is outside the experimental CUDA subset."""

    def __init__(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        self.diagnostics = diagnostics
        summary = "; ".join(
            f"{item.code}: {item.message}" for item in diagnostics
        )
        super().__init__(summary or "CUDA lowering failed")


def emit_cuda(source: str, function_name: str) -> str:
    """Compile and emit a deterministic CUDA module for one named function."""
    return emit_cuda_module(compile_source(source), function_name)


def emit_cuda_module(module: IRModule, function_name: str) -> str:
    """Emit one explicit host wrapper and kernel for a supported tensor function."""
    try:
        function = module.function(function_name)
    except KeyError as error:
        span = module.functions[0].span if module.functions else None
        if span is None:
            raise ValueError(f"Unknown Aether function {function_name!r}") from error
        raise CudaLoweringError(
            (
                Diagnostic(
                    "E_CUDA_FUNCTION",
                    f"Unknown Aether function {function_name!r}",
                    span,
                ),
            )
        ) from error

    optimized = fuse_elementwise_tensors(module)
    function = optimized.function(function_name)
    _validate_function(function)
    returned = function.body[0].expression
    if returned is None:
        _fail(function, "E_CUDA_RETURN", "Expected a tensor return expression")
    root = returned.children[0] if returned.kind == "fused_tensor" else returned
    parameter_names = {parameter.name for parameter in function.parameters}
    if not _identifiers_are_parameters(root, parameter_names):
        _fail(
            function,
            "E_CUDA_LOCAL",
            "CUDA v0 requires the fused expression to refer only to parameters",
        )

    dimensions = function.return_type.dimensions
    count = 1
    for dimension in dimensions:
        count *= dimension
        if count > _MAX_CUDA_ELEMENTS:
            _fail(
                function,
                "E_CUDA_SIZE",
                "Tensor element count exceeds the supported host size",
            )
    shape_name = "x".join(str(dimension) for dimension in dimensions)
    tensor_type = f"aether_tensor_f32_{shape_name}"
    grid_blocks = min(
        (count + _CUDA_BLOCK_SIZE - 1) // _CUDA_BLOCK_SIZE,
        65535,
    )
    kernel_name = "aether_kernel_" + function.name.encode("utf-8").hex()
    wrapper_name = "aether_cuda_" + function.name.encode("utf-8").hex()
    expression_code = _emit_expression(root, function, parameter_names)
    kernel_parameters = []
    wrapper_parameters = []
    device_declarations = []
    transfers = []
    cleanup = []
    for parameter in function.parameters:
        encoded = _encoded_identifier(parameter.name)
        if parameter.type.name == "Tensor":
            kernel_parameters.append(f"const float *{encoded}")
            wrapper_parameters.append(f"const {tensor_type} *{encoded}")
            device_name = f"{encoded}_device"
            device_declarations.append(f"    float *{device_name} = NULL;")
            transfers.extend(
                [
                    f"    if (cudaMalloc((void **)&{device_name}, "
                    f"{count} * sizeof(float)) != cudaSuccess) {{",
                    "        aether_status = AETHER_CUDA_RUNTIME_ERROR;",
                    "        goto cleanup;",
                    "    }",
                    f"    if (cudaMemcpy({device_name}, {encoded}->data, "
                    f"{count} * sizeof(float), cudaMemcpyHostToDevice) "
                    "!= cudaSuccess) {",
                    "        aether_status = AETHER_CUDA_RUNTIME_ERROR;",
                    "        goto cleanup;",
                    "    }",
                ]
            )
            cleanup.append(
                f"    if ({device_name} != NULL && cudaFree({device_name}) "
                "!= cudaSuccess && aether_status == AETHER_CUDA_SUCCESS)"
            )
            cleanup.append(
                "        aether_status = AETHER_CUDA_RUNTIME_ERROR;"
            )
        else:
            kernel_parameters.append(f"float {encoded}")
            wrapper_parameters.append(f"float {encoded}")

    output_device = "aether_output_device"
    device_declarations.append(f"    float *{output_device} = NULL;")
    wrapper_parameters.append(f"{tensor_type} *aether_output")
    transfers.extend(
        [
            f"    if (cudaMalloc((void **)&{output_device}, "
            f"{count} * sizeof(float)) != cudaSuccess) {{",
            "        aether_status = AETHER_CUDA_RUNTIME_ERROR;",
            "        goto cleanup;",
            "    }",
            f"    {kernel_name}<<<{grid_blocks}, {_CUDA_BLOCK_SIZE}>>>"
            f"({', '.join(_kernel_argument(parameter) for parameter in function.parameters)}"
            f"{', ' if function.parameters else ''}{output_device});",
            "    if (cudaGetLastError() != cudaSuccess",
            "        || cudaDeviceSynchronize() != cudaSuccess) {",
            "        aether_status = AETHER_CUDA_RUNTIME_ERROR;",
            "        goto cleanup;",
            "    }",
            f"    if (cudaMemcpy(aether_output->data, {output_device}, "
            f"{count} * sizeof(float), cudaMemcpyDeviceToHost) "
            "!= cudaSuccess) {",
            "        aether_status = AETHER_CUDA_RUNTIME_ERROR;",
            "        goto cleanup;",
            "    }",
        ]
    )
    cleanup.append(
        f"    if ({output_device} != NULL && cudaFree({output_device}) "
        "!= cudaSuccess && aether_status == AETHER_CUDA_SUCCESS)"
    )
    cleanup.append("        aether_status = AETHER_CUDA_RUNTIME_ERROR;")
    null_checks = [
        f"{_encoded_identifier(parameter.name)} == NULL"
        for parameter in function.parameters
        if parameter.type.name == "Tensor"
    ]
    null_checks.append("aether_output == NULL")
    wrapper_arguments = ", ".join(wrapper_parameters)
    kernel_arguments = ", ".join(kernel_parameters)
    lines = [
        "#include <cuda_runtime.h>",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        "#define AETHER_CUDA_SUCCESS 0",
        "#define AETHER_CUDA_NO_DEVICE 1",
        "#define AETHER_CUDA_INVALID_ARGUMENT 2",
        "#define AETHER_CUDA_RUNTIME_ERROR 3",
        "",
        f"struct {tensor_type} {{ float data[{count}]; }};",
        "",
        f"__global__ void {kernel_name}({kernel_arguments}, float *aether_output) {{",
        "    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;",
        f"         index < {count}; index += (size_t)blockDim.x * gridDim.x) {{",
        f"        aether_output[index] = {expression_code};",
        "    }",
        "}",
        "",
        f'extern "C" int {wrapper_name}({wrapper_arguments}) {{',
        "    int aether_status = AETHER_CUDA_SUCCESS;",
        "    int aether_device_count = 0;",
        f"    if ({' || '.join(null_checks)})",
        "        return AETHER_CUDA_INVALID_ARGUMENT;",
        "    cudaError_t aether_device_status =",
        "        cudaGetDeviceCount(&aether_device_count);",
        "    if (aether_device_status == cudaErrorNoDevice",
        "        || (aether_device_status == cudaSuccess",
        "            && aether_device_count < 1))",
        "        return AETHER_CUDA_NO_DEVICE;",
        "    if (aether_device_status != cudaSuccess)",
        "        return AETHER_CUDA_RUNTIME_ERROR;",
        *device_declarations,
        *transfers,
        "cleanup:",
        *cleanup,
        "    return aether_status;",
        "}",
        "",
    ]
    return "\n".join(lines)


def _validate_function(function: IRFunction) -> None:
    if function.external_abi is not None or function.required_effects:
        _fail(function, "E_CUDA_EFFECT", "Effects and external functions are unsupported")
    if (
        function.return_type.name != "Tensor"
        or function.return_type.element_type is None
        or function.return_type.element_type.name != "f32"
        or not function.return_type.dimensions
        or _has_unit(function.return_type)
    ):
        _fail(function, "E_CUDA_RETURN_TYPE", "CUDA v0 returns a unitless f32 tensor")
    if (
        len(function.body) != 1
        or function.body[0].kind != "return"
        or function.body[0].expression is None
        or function.body[0].expression.kind
        not in {"fused_tensor", "binary", "unary", "group"}
    ):
        _fail(
            function,
            "E_CUDA_BODY",
            "CUDA v0 supports a pure fused elementwise tensor return only",
        )
    tensor_parameters = 0
    for parameter in function.parameters:
        if parameter.type.name == "Tensor":
            tensor_parameters += 1
            if (
                parameter.type.element_type != function.return_type.element_type
                or parameter.type.dimensions != function.return_type.dimensions
                or _has_unit(parameter.type)
            ):
                _fail(
                    function,
                    "E_CUDA_PARAMETER",
                    "Tensor parameters must be unitless f32 with the return shape",
                )
        elif parameter.type.name != "f32" or _has_unit(parameter.type):
            _fail(
                function,
                "E_CUDA_PARAMETER",
                "CUDA v0 scalar parameters must be unitless f32",
            )
    if tensor_parameters == 0:
        _fail(function, "E_CUDA_PARAMETER", "At least one tensor parameter is required")
    count = 1
    for dimension in function.return_type.dimensions:
        if dimension < 1:
            _fail(function, "E_CUDA_SIZE", "Tensor dimensions must be positive")
        count *= dimension
        if count > _MAX_CUDA_ELEMENTS:
            _fail(function, "E_CUDA_SIZE", "Tensor element count is too large")


def _has_unit(type_: AetherType) -> bool:
    return bool(
        type_.unit
        or type_.unit_scale != 1
        or type_.unit_offset
        or type_.unit_symbol
        or type_.unit_kind
        or type_.unit_pi_power
    )


def _fail(function: IRFunction, code: str, message: str) -> None:
    raise CudaLoweringError(
        (Diagnostic(code, message, function.span),)
    )


def _encoded_identifier(name: str) -> str:
    return "aether_v_" + name.encode("utf-8").hex()


def _kernel_argument(parameter) -> str:
    encoded = _encoded_identifier(parameter.name)
    return f"{encoded}_device" if parameter.type.name == "Tensor" else encoded


def _identifiers_are_parameters(
    expression: IRExpression,
    parameter_names: set[str],
) -> bool:
    if expression.kind == "identifier" and expression.value not in parameter_names:
        return False
    return all(
        _identifiers_are_parameters(child, parameter_names)
        for child in expression.children
    )


def _emit_expression(
    expression: IRExpression,
    function: IRFunction,
    parameter_names: set[str],
) -> str:
    if expression.kind == "literal":
        value = expression.value
        if expression.type.name != "f32":
            _fail(function, "E_CUDA_EXPRESSION", "Only f32 literals are supported")
        return value if value.lower().endswith("f") else f"{value}f"
    if expression.kind == "identifier":
        if expression.value not in parameter_names:
            _fail(
                function,
                "E_CUDA_LOCAL",
                "CUDA v0 expressions may only reference parameters",
            )
        encoded = _encoded_identifier(expression.value)
        if expression.type.name == "Tensor":
            return f"{encoded}[index]"
        if expression.type.name == "f32":
            return encoded
        _fail(function, "E_CUDA_EXPRESSION", "Only f32 parameters are supported")
    if expression.kind == "group":
        return f"({_emit_expression(expression.children[0], function, parameter_names)})"
    if expression.kind == "unary" and expression.value in {"+", "-"}:
        operand = _emit_expression(
            expression.children[0],
            function,
            parameter_names,
        )
        return f"static_cast<float>(({expression.value}{operand}))"
    if expression.kind == "binary" and expression.value in {"+", "-", "*"}:
        left = _emit_expression(expression.children[0], function, parameter_names)
        right = _emit_expression(expression.children[1], function, parameter_names)
        return f"static_cast<float>(({left} {expression.value} {right}))"
    _fail(
        function,
        "E_CUDA_EXPRESSION",
        f"Unsupported CUDA v0 expression kind {expression.kind!r}",
    )
