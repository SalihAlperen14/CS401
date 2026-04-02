"""
bug_strategies.py - Deterministic AST-based bug injection strategies.

Each strategy is a callable:
    inject(source: str) -> str | None

Returns the mutated source, or None if no suitable injection site was found.
"""

from __future__ import annotations

import ast
import random
import textwrap
from typing import Callable, Dict, List, Optional

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_parse(source: str) -> Optional[ast.Module]:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _unparse(tree: ast.AST) -> str:
    return ast.unparse(tree)


# ── Strategy: off-by-one ──────────────────────────────────────────────────────

class _OffByOneMutator(ast.NodeTransformer):
    """Add or subtract 1 from a randomly chosen integer constant."""

    def __init__(self):
        self.candidates: List[ast.Constant] = []
        self._target: Optional[ast.Constant] = None

    def _collect(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                self.candidates.append(node)

    def choose(self, tree: ast.AST) -> bool:
        self._collect(tree)
        if not self.candidates:
            return False
        self._target = random.choice(self.candidates)
        return True

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if node is self._target and isinstance(node.value, int):
            delta = random.choice([-1, 1])
            return ast.Constant(value=node.value + delta)
        return node


def inject_off_by_one(source: str) -> Optional[str]:
    tree = _safe_parse(source)
    if tree is None:
        return None
    mutator = _OffByOneMutator()
    if not mutator.choose(tree):
        return None
    new_tree = mutator.visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return _unparse(new_tree)
    except Exception:
        return None


# ── Strategy: wrong operator ──────────────────────────────────────────────────

_COMPARE_SWAPS: Dict[type, type] = {
    ast.Lt:    ast.LtE,
    ast.LtE:   ast.Lt,
    ast.Gt:    ast.GtE,
    ast.GtE:   ast.Gt,
    ast.Eq:    ast.NotEq,
    ast.NotEq: ast.Eq,
}

_BOOL_SWAPS: Dict[type, type] = {
    ast.And: ast.Or,
    ast.Or:  ast.And,
}

_ARITH_SWAPS: Dict[type, type] = {
    ast.Add:  ast.Sub,
    ast.Sub:  ast.Add,
    ast.Mult: ast.FloorDiv,
}


class _OperatorMutator(ast.NodeTransformer):
    def __init__(self, swap_map: Dict[type, type]):
        self.swap_map = swap_map
        self.candidates: List[ast.AST] = []
        self._target: Optional[ast.AST] = None

    def _collect(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if type(node) in self.swap_map:
                self.candidates.append(node)

    def choose(self, tree: ast.AST) -> bool:
        self._collect(tree)
        if not self.candidates:
            return False
        self._target = random.choice(self.candidates)
        return True

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if node is self._target:
            new_cls = self.swap_map[type(node)]
            return new_cls()
        return super().generic_visit(node)


def _operator_inject(source: str, swap_map: Dict[type, type]) -> Optional[str]:
    tree = _safe_parse(source)
    if tree is None:
        return None
    mutator = _OperatorMutator(swap_map)
    if not mutator.choose(tree):
        return None
    new_tree = mutator.visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return _unparse(new_tree)
    except Exception:
        return None


def inject_wrong_operator(source: str) -> Optional[str]:
    return _operator_inject(source, _COMPARE_SWAPS)


def inject_logic_inversion(source: str) -> Optional[str]:
    return _operator_inject(source, _BOOL_SWAPS)


def inject_wrong_arithmetic(source: str) -> Optional[str]:
    return _operator_inject(source, _ARITH_SWAPS)


# ── Strategy: missing return ──────────────────────────────────────────────────

class _MissingReturnMutator(ast.NodeTransformer):
    """Remove the return statement from a randomly chosen function."""

    def __init__(self):
        self.candidates: List[ast.FunctionDef] = []
        self._target: Optional[ast.FunctionDef] = None

    def _collect(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Only functions that have an explicit return with a value
                has_return = any(
                    isinstance(n, ast.Return) and n.value is not None
                    for n in ast.walk(node)
                )
                if has_return:
                    self.candidates.append(node)

    def choose(self, tree: ast.AST) -> bool:
        self._collect(tree)
        if not self.candidates:
            return False
        self._target = random.choice(self.candidates)
        return True

    def visit_Return(self, node: ast.Return) -> Optional[ast.AST]:
        # We're inside the targeted function when the transformer visits it;
        # replace the return with a bare `pass` expression
        return ast.Pass()


def inject_missing_return(source: str) -> Optional[str]:
    tree = _safe_parse(source)
    if tree is None:
        return None
    mutator = _MissingReturnMutator()
    if not mutator.choose(tree):
        return None

    # Only mutate inside the chosen function
    class _Scoped(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            if node is mutator._target:
                return mutator.visit(node)
            return node

    new_tree = _Scoped().visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return _unparse(new_tree)
    except Exception:
        return None


# ── Strategy: wrong variable name ─────────────────────────────────────────────

class _WrongVariableMutator(ast.NodeTransformer):
    """Swap two same-scope local variable names."""

    def __init__(self):
        self.pair: Optional[tuple[str, str]] = None

    def choose(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [
                    n.id for n in ast.walk(node)
                    if isinstance(n, ast.Name)
                    and isinstance(n.ctx, ast.Store)
                ]
                unique = list(set(names))
                if len(unique) >= 2:
                    random.shuffle(unique)
                    self.pair = (unique[0], unique[1])
                    return True
        return False

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.pair:
            a, b = self.pair
            if node.id == a:
                return ast.Name(id=b, ctx=node.ctx)
            if node.id == b:
                return ast.Name(id=a, ctx=node.ctx)
        return node


def inject_wrong_variable(source: str) -> Optional[str]:
    tree = _safe_parse(source)
    if tree is None:
        return None
    mutator = _WrongVariableMutator()
    if not mutator.choose(tree):
        return None
    new_tree = mutator.visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return _unparse(new_tree)
    except Exception:
        return None


# ── Strategy: wrong constant ──────────────────────────────────────────────────

def inject_wrong_constant(source: str) -> Optional[str]:
    """Flip a boolean constant (True <-> False) or negate a string sentinel."""
    tree = _safe_parse(source)
    if tree is None:
        return None

    bools = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, bool)
    ]
    if not bools:
        return None

    target = random.choice(bools)

    class _Flip(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            if node is target:
                return ast.Constant(value=not node.value)
            return node

    new_tree = _Flip().visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return _unparse(new_tree)
    except Exception:
        return None


# ── Strategy: missing condition (remove an `if` guard) ───────────────────────

class _MissingConditionMutator(ast.NodeTransformer):
    """Remove the condition of an `if` block, always executing the body."""

    def __init__(self):
        self.candidates: List[ast.If] = []
        self._target: Optional[ast.If] = None

    def _collect(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and node.body:
                self.candidates.append(node)

    def choose(self, tree: ast.AST) -> bool:
        self._collect(tree)
        if not self.candidates:
            return False
        self._target = random.choice(self.candidates)
        return True

    def visit_If(self, node: ast.If) -> ast.AST:
        if node is self._target:
            # Replace the entire if-else with just the body statements
            return node.body  # type: ignore[return-value]
        return self.generic_visit(node)


def inject_missing_condition(source: str) -> Optional[str]:
    tree = _safe_parse(source)
    if tree is None:
        return None

    mutator = _MissingConditionMutator()
    if not mutator.choose(tree):
        return None

    # ast.NodeTransformer can return a list from visit_If, which requires
    # special handling via fix_missing_locations + flattening
    class _FlatTransformer(ast.NodeTransformer):
        def visit_If(self, node: ast.If) -> ast.AST:
            if node is mutator._target:
                return node.body  # type: ignore[return-value]
            return self.generic_visit(node)

    new_tree = _FlatTransformer().visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return _unparse(new_tree)
    except Exception:
        return None


# ── Registry ──────────────────────────────────────────────────────────────────

STRATEGIES: Dict[str, Callable[[str], Optional[str]]] = {
    "off_by_one":       inject_off_by_one,
    "wrong_operator":   inject_wrong_operator,
    "logic_inversion":  inject_logic_inversion,
    "wrong_arithmetic": inject_wrong_arithmetic,
    "missing_return":   inject_missing_return,
    "wrong_variable":   inject_wrong_variable,
    "wrong_constant":   inject_wrong_constant,
    "missing_condition": inject_missing_condition,
}
