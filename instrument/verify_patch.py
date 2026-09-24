#!/usr/bin/env python3
"""Prove that the instrumented runner is the baseline plus probes only.

The instrumentation only ever *adds* ``with pi_scope("...")`` statements
(plus the single ``pi_prepare_input(...)`` wrapper around the LiteProfiler
``record_function_or_nullcontext("prepare input")`` scope).  Therefore, after
mechanically removing those inserted statements from the AST of the
instrumented file, the AST must be *identical* to the baseline AST.

That is a much stronger statement than "it compiles": it rules out accidental
re-indentation bugs, dropped statements, changed literals and changed control
flow.

Usage::

    python3 verify_patch.py BASELINE.py INSTRUMENTED.py

Exit code 0 = equivalent, 1 = mismatch (a unified AST diff is printed).
"""

from __future__ import annotations

import ast
import difflib
import sys


def _is_pi_scope(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "pi_scope"
    )


def _is_pi_prepare_input(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "pi_prepare_input"
    )


class _Strip(ast.NodeTransformer):
    """Remove probe wrappers, restoring the baseline statement structure."""

    def __init__(self) -> None:
        # Instance state, deliberately not class attributes: a mutable class
        # attribute would leak the report of one file into the next.
        self.removed_scopes: list[str] = []
        self.removed_calls: list[str] = []

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module == "vllm_ascend.worker.pi_subscope":
            self.removed_calls.append("import vllm_ascend.worker.pi_subscope")
            return None
        return node

    def visit_Expr(self, node: ast.Expr):
        self.generic_visit(node)
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "pi_step_info"
        ):
            self.removed_calls.append("pi_step_info(...)")
            return None
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "pi_note"
        ):
            self.removed_calls.append("pi_note(...)")
            return None
        return node

    def visit_Try(self, node: ast.Try):
        """Unwrap diagnostic ``try: ... except Exception: pass`` scaffolding.

        Only a try-block whose body is *entirely* probe scaffolding disappears.
        Baseline try-blocks (upstream code as well as the LiteProfiler patch)
        are left alone: their bodies survive the visit, so ``out`` is non-empty
        and the node is returned with its original handlers.
        """
        if (
            len(node.handlers) == 1
            and node.handlers[0].type is not None
            and getattr(node.handlers[0].type, "id", None) == "Exception"
            and not node.handlers[0].name
            and len(node.handlers[0].body) == 1
            and isinstance(node.handlers[0].body[0], ast.Pass)
            and not node.orelse
            and not node.finalbody
        ):
            out: list[ast.stmt] = []
            for stmt in node.body:
                visited = self.visit(stmt)
                if visited is None:
                    continue
                if isinstance(visited, list):
                    out.extend(visited)
                else:
                    out.append(visited)
            if not out:
                self.removed_calls.append(
                    "try: <probe only> except Exception: pass"
                )
                return None
            # A baseline try-block: keep it, but with the probe statements gone.
            node.body = out
            return node
        self.generic_visit(node)
        return node

    def visit_With(self, node: ast.With):
        self.generic_visit(node)
        kept: list[ast.withitem] = []
        for item in node.items:
            expr = item.context_expr
            if _is_pi_scope(expr):
                if isinstance(expr, ast.Call) and expr.args:
                    self.removed_scopes.append(ast.unparse(expr.args[0]))
                continue
            if _is_pi_prepare_input(expr):
                inner = expr.args[0]
                if item.optional_vars is not None:
                    raise SystemExit(
                        "pi_prepare_input() must not be used with 'as' at "
                        f"line {getattr(node, 'lineno', '?')}"
                    )
                kept.append(
                    ast.withitem(
                        context_expr=inner,
                        optional_vars=None,
                    )
                )
                continue
            kept.append(item)
        if not kept:
            return node.body
        node.items = kept
        return node


def normalise(path: str, *, strip: bool) -> str:
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    if strip:
        tree = _Strip().visit(tree)
        ast.fix_missing_locations(tree)
    return ast.dump(tree, indent=1)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    base_path, inst_path = argv[1], argv[2]
    base = normalise(base_path, strip=False)
    inst = normalise(inst_path, strip=True)
    if base == inst:
        print(f"EQUIVALENT: {inst_path} == {base_path} + probes")
        return 0
    print(f"MISMATCH between {base_path} and {inst_path} (after stripping probes)")
    diff = difflib.unified_diff(
        base.splitlines(), inst.splitlines(), fromfile="baseline", tofile="patched"
    )
    for i, line in enumerate(diff):
        if i > 200:
            print("... (truncated)")
            break
        print(line)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
