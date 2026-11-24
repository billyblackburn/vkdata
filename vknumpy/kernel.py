"""vknumpy.kernel: Python-syntax Vulkan GPU compute kernels via AST -> GLSL 450 -> SPIR-V."""

from __future__ import annotations

import ast
import inspect
import shutil
import struct
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as _np

# --- Built-in kernel intrinsics (recognized by the AST transpiler) ---
_INTRINSICS: Dict[str, Tuple[str, str]] = {
    "global_id_x":  ("int(gl_GlobalInvocationID.x)", "int"),
    "global_id_y":  ("int(gl_GlobalInvocationID.y)", "int"),
    "global_id_z":  ("int(gl_GlobalInvocationID.z)", "int"),
    "local_id_x":   ("int(gl_LocalInvocationID.x)",  "int"),
    "local_id_y":   ("int(gl_LocalInvocationID.y)",  "int"),
    "local_id_z":   ("int(gl_LocalInvocationID.z)",  "int"),
    "group_id_x":   ("int(gl_WorkGroupID.x)",        "int"),
    "group_id_y":   ("int(gl_WorkGroupID.y)",        "int"),
    "group_id_z":   ("int(gl_WorkGroupID.z)",        "int"),
    "group_size_x": ("int(gl_WorkGroupSize.x)",      "int"),
    "group_size_y": ("int(gl_WorkGroupSize.y)",      "int"),
    "group_size_z": ("int(gl_WorkGroupSize.z)",      "int"),
}

_MATH_FUNCS = {"exp", "log", "sqrt", "sin", "cos", "tan", "abs", "min", "max", "floor", "ceil", "pow", "clamp"}

_GLSL_RESERVED = {
    "in", "out", "inout", "uniform", "buffer", "shared", "layout", "main",
    "input", "output", "sample", "patch", "centroid", "flat", "smooth", "noperspective",
    "vec2", "vec3", "vec4", "mat2", "mat3", "mat4", "uint", "bool", "void",
}


def _safe_id(name: str) -> str:
    return f"_{name}" if name in _GLSL_RESERVED else name


def global_id_x() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def global_id_y() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def global_id_z() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def local_id_x() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def local_id_y() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def local_id_z() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def group_id_x() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def group_id_y() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def group_id_z() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def group_size_x() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def group_size_y() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def group_size_z() -> int: raise RuntimeError("Call inside a @vnp.kernel function")
def barrier() -> None: raise RuntimeError("Call inside a @vnp.kernel function")
def shared_array(size: int, dtype: str = "float"): raise RuntimeError("Call inside a @vnp.kernel function")


def _binop_str(op: ast.AST) -> str:
    return {
        ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
        ast.FloorDiv: "/", ast.Mod: "%", ast.BitAnd: "&", ast.BitOr: "|",
        ast.BitXor: "^", ast.LShift: "<<", ast.RShift: ">>",
    }[type(op)]


def _cmpop_str(op: ast.AST) -> str:
    return {
        ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<",
        ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    }[type(op)]


class _GLSLTranspiler:
    """Translates a restricted Python AST function into a Vulkan GLSL 450 compute shader."""

    def __init__(self, func_def: ast.FunctionDef, local_size: Tuple[int, int, int]) -> None:
        self.func_def = func_def
        self.local_size = local_size
        self.buf_params: List[Tuple[str, str]] = []     # (name, glsl_elem_type)
        self.scalar_params: List[Tuple[str, str]] = []  # (name, glsl_type)
        self.shared_decls: List[str] = []
        self.var_types: Dict[str, str] = {}             # local variable -> "int" | "float" | "bool"
        self.body_lines: List[str] = []

        for arg in func_def.args.args:
            ann = self._ann_name(arg.annotation)
            if ann in ("int", "int32"):
                self.scalar_params.append((arg.arg, "int"))
                self.var_types[arg.arg] = "int"
            elif ann in ("float", "float32"):
                self.scalar_params.append((arg.arg, "float"))
                self.var_types[arg.arg] = "float"
            elif ann in ("int_buf", "i32"):
                self.buf_params.append((arg.arg, "int"))
            else:
                # Unannotated or float_buf -> float32 Vulkan SSBO
                self.buf_params.append((arg.arg, "float"))

        self.scalar_names = {name for name, _ in self.scalar_params}
        self.buf_types = {name: et for name, et in self.buf_params}

    @staticmethod
    def _ann_name(ann: Optional[ast.AST]) -> Optional[str]:
        if ann is None:
            return None
        if isinstance(ann, ast.Name):
            return ann.id
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            return ann.value
        return None

    def transpile(self) -> str:
        for stmt in self.func_def.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue  # skip docstring
            self._emit_stmt(stmt, indent=1)

        lx, ly, lz = self.local_size
        lines = [
            "#version 450",
            f"layout(local_size_x = {lx}, local_size_y = {ly}, local_size_z = {lz}) in;",
            "",
        ]
        for idx, (bname, etype) in enumerate(self.buf_params):
            gb = _safe_id(bname)
            lines.append(f"layout(set = 0, binding = {idx}) buffer Buf_{gb} {{ {etype} {gb}[]; }};")
        if self.scalar_params:
            lines.append("")
            lines.append("layout(push_constant) uniform PC {")
            for sname, stype in self.scalar_params:
                lines.append(f"    {stype} {_safe_id(sname)};")
            lines.append("} pc;")
        if self.shared_decls:
            lines.append("")
            lines.extend(self.shared_decls)
        lines.append("")
        lines.append("void main() {")
        lines.extend(self.body_lines)
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _emit_stmt(self, stmt: ast.AST, indent: int) -> None:
        pad = "    " * indent

        # shared_array declaration: tile = shared_array(256)
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "shared_array"
        ):
            name = stmt.targets[0].id
            sz = int(ast.literal_eval(stmt.value.args[0]))
            etype = "float"
            if len(stmt.value.args) > 1:
                etype = str(ast.literal_eval(stmt.value.args[1]))
            self.shared_decls.append(f"shared {etype} {name}[{sz}];")
            self.buf_types[name] = etype
            return

        if isinstance(stmt, ast.Assign):
            assert len(stmt.targets) == 1, "Only single-target assignment is supported"
            tgt = stmt.targets[0]
            rhs_code, rhs_t = self._expr(stmt.value)
            if isinstance(tgt, ast.Name):
                vname = tgt.id
                if vname not in self.var_types:
                    self.var_types[vname] = rhs_t
                    self.body_lines.append(f"{pad}{rhs_t} {vname} = {rhs_code};")
                else:
                    self.body_lines.append(f"{pad}{vname} = {rhs_code};")
            elif isinstance(tgt, ast.Subscript):
                lhs_code, _ = self._expr(tgt)
                self.body_lines.append(f"{pad}{lhs_code} = {rhs_code};")
            else:
                raise SyntaxError(f"Unsupported assignment target: {ast.dump(tgt)}")

        elif isinstance(stmt, ast.AugAssign):
            lhs_code, _ = self._expr(stmt.target)
            rhs_code, _ = self._expr(stmt.value)
            op = _binop_str(stmt.op)
            self.body_lines.append(f"{pad}{lhs_code} {op}= {rhs_code};")

        elif isinstance(stmt, ast.If):
            cond, _ = self._expr(stmt.test)
            self.body_lines.append(f"{pad}if ({cond}) {{")
            for s in stmt.body:
                self._emit_stmt(s, indent + 1)
            if stmt.orelse:
                self.body_lines.append(f"{pad}}} else {{")
                for s in stmt.orelse:
                    self._emit_stmt(s, indent + 1)
            self.body_lines.append(f"{pad}}}")

        elif isinstance(stmt, ast.For):
            if not (
                isinstance(stmt.target, ast.Name)
                and isinstance(stmt.iter, ast.Call)
                and isinstance(stmt.iter.func, ast.Name)
                and stmt.iter.func.id == "range"
            ):
                raise SyntaxError("Only `for i in range(...)` loops are supported in @vnp.kernel")
            ivar = stmt.target.id
            rargs = stmt.iter.args
            if len(rargs) == 1:
                start_c, stop_c, step_c = "0", self._expr(rargs[0])[0], "1"
            elif len(rargs) == 2:
                start_c, stop_c, step_c = self._expr(rargs[0])[0], self._expr(rargs[1])[0], "1"
            else:
                start_c, stop_c, step_c = self._expr(rargs[0])[0], self._expr(rargs[1])[0], self._expr(rargs[2])[0]
            self.var_types[ivar] = "int"
            self.body_lines.append(f"{pad}for (int {ivar} = {start_c}; {ivar} < {stop_c}; {ivar} += {step_c}) {{")
            for s in stmt.body:
                self._emit_stmt(s, indent + 1)
            self.body_lines.append(f"{pad}}}")

        elif isinstance(stmt, ast.While):
            cond, _ = self._expr(stmt.test)
            self.body_lines.append(f"{pad}while ({cond}) {{")
            for s in stmt.body:
                self._emit_stmt(s, indent + 1)
            self.body_lines.append(f"{pad}}}")

        elif isinstance(stmt, ast.Return):
            self.body_lines.append(f"{pad}return;")

        elif isinstance(stmt, ast.Break):
            self.body_lines.append(f"{pad}break;")

        elif isinstance(stmt, ast.Continue):
            self.body_lines.append(f"{pad}continue;")

        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "barrier":
                self.body_lines.append(f"{pad}barrier();")
            else:
                c, _ = self._expr(stmt.value)
                self.body_lines.append(f"{pad}{c};")
        else:
            raise SyntaxError(f"Unsupported statement in @vnp.kernel: {ast.dump(stmt)}")

    def _expr(self, node: ast.AST) -> Tuple[str, str]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return ("true" if node.value else "false"), "bool"
            if isinstance(node.value, int):
                return str(node.value), "int"
            if isinstance(node.value, float):
                return repr(float(node.value)), "float"
            raise SyntaxError(f"Unsupported constant: {node.value!r}")

        if isinstance(node, ast.Name):
            if node.id in self.scalar_names:
                return f"pc.{_safe_id(node.id)}", self.var_types[node.id]
            return _safe_id(node.id), self.var_types.get(node.id, "float")

        if isinstance(node, ast.Subscript):
            raw_name = node.value.id if isinstance(node.value, ast.Name) else ""
            arr_code, _ = self._expr(node.value)
            idx_code, _ = self._expr(node.slice)
            etype = self.buf_types.get(raw_name, "float")
            return f"{arr_code}[{idx_code}]", etype

        if isinstance(node, ast.UnaryOp):
            val, vt = self._expr(node.operand)
            if isinstance(node.op, ast.USub):
                return f"(-{val})", vt
            if isinstance(node.op, ast.Not):
                return f"(!({val}))", "bool"
            raise SyntaxError(f"Unsupported unary op: {type(node.op)}")

        if isinstance(node, ast.BinOp):
            lc, lt = self._expr(node.left)
            rc, rt = self._expr(node.right)
            op = _binop_str(node.op)
            if isinstance(node.op, ast.FloorDiv):
                return f"({lc} / {rc})", "int"
            res_t = "float" if ("float" in (lt, rt) or isinstance(node.op, ast.Div)) else "int"
            if res_t == "float":
                if lt == "int":
                    lc = f"float({lc})"
                if rt == "int":
                    rc = f"float({rc})"
            return f"({lc} {op} {rc})", res_t

        if isinstance(node, ast.BoolOp):
            op = " && " if isinstance(node.op, ast.And) else " || "
            parts = [f"({self._expr(v)[0]})" for v in node.values]
            return f"({op.join(parts)})", "bool"

        if isinstance(node, ast.Compare):
            lc, _ = self._expr(node.left)
            comps = []
            cur = lc
            for op, comp in zip(node.ops, node.comparators):
                rc, _ = self._expr(comp)
                comps.append(f"({cur} {_cmpop_str(op)} {rc})")
                cur = rc
            return f"({' && '.join(comps)})", "bool"

        if isinstance(node, ast.IfExp):
            cond, _ = self._expr(node.test)
            tc, tt = self._expr(node.body)
            fc, ft = self._expr(node.orelse)
            res_t = "float" if "float" in (tt, ft) else tt
            return f"(({cond}) ? ({tc}) : ({fc}))", res_t

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname in _INTRINSICS:
                return _INTRINSICS[fname]
            if fname in ("float", "float32"):
                ac, _ = self._expr(node.args[0])
                return f"float({ac})", "float"
            if fname in ("int", "int32"):
                ac, _ = self._expr(node.args[0])
                return f"int({ac})", "int"
            if fname in _MATH_FUNCS:
                args_c = [self._expr(a)[0] for a in node.args]
                return f"{fname}({', '.join(args_c)})", "float"
            raise SyntaxError(f"Unsupported function call '{fname}' in @vnp.kernel")

        raise SyntaxError(f"Unsupported expression in @vnp.kernel: {ast.dump(node)}")


def _compile_glsl_to_spirv(glsl_src: str) -> bytes:
    """Compile GLSL 450 compute shader source to SPIR-V bytes via glslangValidator."""
    compiler = shutil.which("glslangValidator") or shutil.which("glslang")
    if compiler is None:
        raise RuntimeError("glslangValidator not found on PATH (install `glslang` in pixi.toml)")
    with tempfile.TemporaryDirectory() as td:
        src_p = Path(td) / "kernel.comp"
        spv_p = Path(td) / "kernel.spv"
        src_p.write_text(glsl_src, encoding="utf-8")
        proc = subprocess.run(
            [compiler, "-V", "-S", "comp", str(src_p), "-o", str(spv_p)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"GLSL -> SPIR-V compilation failed:\n{proc.stdout}\n{proc.stderr}\n--- Generated GLSL ---\n{glsl_src}"
            )
        return spv_p.read_bytes()


class VulkanKernel:
    """Compiled Python-syntax Vulkan compute kernel (inspectable AST, GLSL, & SPIR-V)."""

    def __init__(self, func, local_size: Tuple[int, int, int] = (64, 1, 1)) -> None:
        self.func = func
        self.name = func.__name__
        self.local_size = tuple(int(x) for x in local_size)
        self.python_source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(self.python_source)
        func_def = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        self.ast_tree = func_def
        self.ast_dump = ast.dump(func_def, indent=2)

        transpiler = _GLSLTranspiler(func_def, self.local_size)
        self.glsl = transpiler.transpile()
        self.buf_params = transpiler.buf_params
        self.scalar_params = transpiler.scalar_params
        self.spirv_bytes = _compile_glsl_to_spirv(self.glsl)

    def __getitem__(self, grid):
        """Return a launcher bound to workgroup grid ``gx`` or ``(gx, gy)`` or ``(gx, gy, gz)``."""
        if isinstance(grid, int):
            g = (grid, 1, 1)
        elif len(grid) == 2:
            g = (int(grid[0]), int(grid[1]), 1)
        elif len(grid) == 3:
            g = (int(grid[0]), int(grid[1]), int(grid[2]))
        else:
            raise ValueError(f"Grid must be int or 2/3-tuple, got {grid!r}")

        def _launch(*args):
            from vkspatial._backend import dispatch_spv_gpu

            expected = len(self.buf_params) + len(self.scalar_params)
            if len(args) != expected:
                raise TypeError(f"{self.name} expects {expected} arguments, got {len(args)}")
            n_b = len(self.buf_params)
            bufs = list(args[:n_b])
            scalars = args[n_b:]
            fmt = "<" + "".join("i" if st == "int" else "f" for _, st in self.scalar_params)
            push_data = struct.pack(fmt, *[int(v) if st == "int" else float(v) for v, (_, st) in zip(scalars, self.scalar_params)]) if scalars else b""
            dispatch_spv_gpu(self.spirv_bytes, bufs, push_data, g)

        return _launch


def kernel(local_size: Tuple[int, int, int] = (64, 1, 1)):
    """Decorator that compiles a Python function via AST -> GLSL 450 -> SPIR-V for zero-copy Vulkan execution."""
    if callable(local_size):
        # Used as @vnp.kernel without parentheses
        return VulkanKernel(local_size, local_size=(64, 1, 1))
    if isinstance(local_size, int):
        ls = (local_size, 1, 1)
    elif len(local_size) == 2:
        ls = (int(local_size[0]), int(local_size[1]), 1)
    else:
        ls = (int(local_size[0]), int(local_size[1]), int(local_size[2]))

    def _decorator(fn):
        return VulkanKernel(fn, local_size=ls)

    return _decorator
