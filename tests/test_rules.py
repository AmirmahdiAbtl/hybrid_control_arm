import numpy as np
import pandas as pd
import pytest

from src.rules import RuleError, apply_rules, evaluate_expression, evaluate_rule

DF = pd.DataFrame(
    {"age": [30, 70, 50, np.nan], "chemo": [1, 0, 1, 1], "grade": ["I", "II", "III", "II"], "n": [0, 2, 5, 1]}
)


def test_comparison_and_logic():
    assert evaluate_rule("age >= 40 and chemo == 1", DF).tolist() == [False, False, True, False]
    assert evaluate_rule("(n >= 1) & (chemo == 1)", DF).tolist() == [False, False, True, True]
    assert evaluate_rule("not (chemo == 1)", DF).tolist() == [False, True, False, False]
    assert evaluate_rule("~(chemo == 1)", DF).tolist() == [False, True, False, False]
    assert evaluate_rule("n * 2 > 3 or chemo == 0", DF).tolist() == [False, True, True, False]


def test_chained_comparison():
    assert evaluate_rule("18 <= age <= 65", DF).tolist() == [True, False, True, False]


def test_in_not_in():
    assert evaluate_rule("grade in ['I', 'II']", DF).tolist() == [True, True, False, True]
    assert evaluate_rule("grade not in ['I']", DF).tolist() == [False, True, True, True]


def test_missing_value_fails_rule():
    assert evaluate_rule("age < 100", DF).tolist() == [True, True, True, False]
    assert evaluate_rule("age < 100 or chemo == 1", DF).iloc[3] == False  # noqa: E712


@pytest.mark.parametrize(
    "rule",
    [
        "__import__('os').system('ls')",
        "age.mean() > 3",
        "df.age > 3",
        "open('x')",
        "age.__class__",
        "unknown > 3",
        "[x for x in age]",
        "lambda: 1",
        "age ** 2 > 4",
        "age // 2 > 4",
        "age > None",
    ],
)
def test_rejects_unsafe_or_unknown(rule):
    with pytest.raises(RuleError):
        evaluate_rule(rule, DF)


def test_rejects_non_boolean():
    with pytest.raises(RuleError):
        evaluate_rule("chemo", DF)
    with pytest.raises(RuleError):
        evaluate_rule("age + 1", DF)
    assert evaluate_rule("chemo == 1", DF).sum() == 3


def test_syntax_error():
    with pytest.raises(RuleError):
        evaluate_rule("age >=", DF)


def test_invalid_rules_reported_not_applied():
    rules = [
        {"id": "a", "rule": "n >= 1", "text": "x"},
        {"id": "b", "rule": "chemo", "text": "bad"},
        {"id": "c", "rule": "evil()", "text": "bad"},
        {"id": "d", "rule": "chemo == 1", "text": "y"},
    ]
    mask, funnel, invalid = apply_rules(DF, rules)
    assert [i["id"] for i in invalid] == ["b", "c"]
    assert mask.tolist() == [False, False, True, True]
    assert funnel.n_after.tolist() == [4, 3, 2]


def test_derive_expression_ifexp():
    out = evaluate_expression("n if chemo == 1 else -1", DF)
    assert out.tolist() == [0, -1, 5, 1]
    with pytest.raises(RuleError):  # if-expressions only in value mode
        evaluate_rule("1 if chemo == 1 else 0", DF)
