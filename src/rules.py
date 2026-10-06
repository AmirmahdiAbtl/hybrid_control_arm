import ast
import operator

import numpy as np
import pandas as pd

_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


class RuleError(ValueError):
    """Raised when a rule or expression is not allowed or cannot be evaluated."""


def apply_rules(df, rules):
    """Apply eligibility rules one after the other and build the funnel.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical patients.
    rules : list of dict
        Each with ``id``, ``text`` and ``rule`` (and optional ``enabled``).

    Returns
    -------
    mask : pandas.Series
        Boolean mask of eligible patients.
    funnel : pandas.DataFrame
        Rows: start, then after each applied rule (``n_after``, ``n_removed``).
    invalid : list of dict
        Rules that could not be evaluated (``id``, ``rule``, ``error``); not applied.
    """
    mask = pd.Series(True, index=df.index)
    rows = [{"step": "all historical patients", "rule": "", "n_after": int(len(df)), "n_removed": 0}]
    invalid = []
    for r in rules:
        if r.get("enabled", True) is False or not r.get("rule"):
            continue
        try:
            m = evaluate_rule(r["rule"], df)
        except RuleError as e:
            invalid.append({"id": r.get("id"), "rule": r["rule"], "error": str(e)})
            continue
        before = int(mask.sum())
        mask &= m
        rows.append(
            {
                "step": f"{r.get('id', '')}: {r.get('text', '')[:70]}",
                "rule": r["rule"],
                "n_after": int(mask.sum()),
                "n_removed": before - int(mask.sum()),
            }
        )
    return mask, pd.DataFrame(rows), invalid


def evaluate_rule(expr, df):
    """Evaluate an eligibility rule on a DataFrame.

    Only column names, constants, comparisons (also chained), ``in`` / ``not in``, ``and`` / ``or`` / ``not``,
    ``& | ~`` and ``+ - * /`` are allowed. A row with a missing value in any referenced column does not satisfy
    the rule.

    Parameters
    ----------
    expr : str
        Rule, TRUE for eligible patients, e.g. ``"nodes >= 1"``.
    df : pandas.DataFrame
        Patients; column names are the variables.

    Returns
    -------
    pandas.Series
        Boolean mask aligned with ``df``.

    Raises
    ------
    RuleError
        If the rule uses a forbidden construct, an unknown column, or is not boolean.
    """
    res, used = _parse_and_evaluate(expr, df, "rule")
    if not _is_boolean(res):
        raise RuleError("rule does not give True/False (compare the column, e.g. chemo == 1)")
    if not isinstance(res, pd.Series):
        res = pd.Series(bool(res), index=df.index)
    ok = ~df[list(used)].isna().any(axis=1) if used else pd.Series(True, index=df.index)
    return (res & ok).astype(bool)


def evaluate_expression(expr, df):
    """Evaluate a whitelisted value expression (used by ``derive`` transforms).

    Same engine as :func:`evaluate_rule`, and additionally ``a if cond else b``.

    Returns
    -------
    pandas.Series
        Values aligned with ``df``.
    """
    res, _ = _parse_and_evaluate(expr, df, "value")
    return res if isinstance(res, pd.Series) else pd.Series(res, index=df.index)


def _parse_and_evaluate(expr, df, mode):
    """Value of ``expr`` and the set of columns it uses; ``mode`` is 'rule' or 'value' (allows if-expressions)."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise RuleError(f"syntax error: {e.msg}") from e
    used = set()
    return _evaluate_node(tree, df, mode, used), used


def _evaluate_node(node, df, mode, used):
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, df, mode, used)
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (bytes, complex)):
            raise RuleError(f"constant {node.value!r} is not allowed")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in df.columns:
            raise RuleError(f"unknown column '{node.id}'")
        used.add(node.id)
        return df[node.id]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return _constant_list(node)
    if isinstance(node, ast.Compare):
        return _comparison(node, df, mode, used)
    if isinstance(node, ast.BoolOp):
        return _boolean_operation(node, df, mode, used)
    if isinstance(node, ast.UnaryOp):
        return _unary_operation(node, df, mode, used)
    if isinstance(node, ast.BinOp):
        return _binary_operation(node, df, mode, used)
    if isinstance(node, ast.IfExp) and mode == "value":
        return _if_expression(node, df, mode, used)
    raise RuleError(f"{type(node).__name__} is not allowed")


def _constant_list(node):
    if not all(isinstance(e, ast.Constant) for e in node.elts):
        raise RuleError("only constants are allowed inside [..] lists")
    return [e.value for e in node.elts]


def _comparison(node, df, mode, used):
    left = _evaluate_node(node.left, df, mode, used)
    out = None
    for op, comparator in zip(node.ops, node.comparators):
        right = _evaluate_node(comparator, df, mode, used)
        if isinstance(op, (ast.In, ast.NotIn)):
            if not isinstance(right, list) or not isinstance(left, pd.Series):
                raise RuleError("'in' needs a column on the left and a [list] on the right")
            res = left.isin(right)
            if isinstance(op, ast.NotIn):
                res = ~res
        elif type(op) in _COMPARISONS:
            try:
                res = _COMPARISONS[type(op)](left, right)
            except TypeError as e:
                raise RuleError(f"cannot compare: {e}") from e
        else:
            raise RuleError(f"operator {type(op).__name__} is not allowed")
        out = res if out is None else (out & res)
        left = right
    return out


def _boolean_operation(node, df, mode, used):
    vals = [_evaluate_node(v, df, mode, used) for v in node.values]
    if not all(_is_boolean(v) for v in vals):
        raise RuleError("'and'/'or' need boolean operands (use e.g. chemo == 1)")
    out = vals[0]
    for v in vals[1:]:
        out = (out & v) if isinstance(node.op, ast.And) else (out | v)
    return out


def _unary_operation(node, df, mode, used):
    v = _evaluate_node(node.operand, df, mode, used)
    if isinstance(node.op, (ast.Not, ast.Invert)):
        if not _is_boolean(v):
            raise RuleError("'not' / '~' need a boolean operand")
        return ~v if isinstance(v, pd.Series) else (not v)
    if isinstance(node.op, ast.USub):
        return -v
    if isinstance(node.op, ast.UAdd):
        return +v
    raise RuleError("unary operator not allowed")


def _binary_operation(node, df, mode, used):
    a, b = _evaluate_node(node.left, df, mode, used), _evaluate_node(node.right, df, mode, used)
    if isinstance(node.op, (ast.BitAnd, ast.BitOr)):
        if not (_is_boolean(a) and _is_boolean(b)):
            raise RuleError("'&' / '|' need boolean operands")
        return a & b if isinstance(node.op, ast.BitAnd) else a | b
    if type(node.op) not in _ARITHMETIC:
        raise RuleError(f"operator {type(node.op).__name__} is not allowed")
    try:
        return _ARITHMETIC[type(node.op)](a, b)
    except TypeError as e:
        raise RuleError(f"cannot compute: {e}") from e


def _if_expression(node, df, mode, used):
    cond = _evaluate_node(node.test, df, mode, used)
    if not _is_boolean(cond):
        raise RuleError("the condition of 'x if cond else y' must be boolean")
    a, b = _evaluate_node(node.body, df, mode, used), _evaluate_node(node.orelse, df, mode, used)
    return pd.Series(np.where(cond, a, b), index=df.index)


def _is_boolean(x):
    if isinstance(x, (bool, np.bool_)):
        return True
    return isinstance(x, pd.Series) and pd.api.types.is_bool_dtype(x)
