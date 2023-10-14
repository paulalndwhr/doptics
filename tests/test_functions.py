import doptics as dp
from doptics.functions import normalize, uniform
from icecream import ic
import pytest


def test_normalize() -> None:
    def f(x): return 1
    span = [0, 1]
    ic(normalize(f, span)(1))
    assert normalize(f, span) == f


@pytest.mark.parametrize("test_function,test_domain,argument,expected", [
    (uniform, [0, 1], 0, 1.0),
    (uniform, [0, 1], 0.5, 1.0),
    (uniform, [0, 1], 1, 1.0),
    (lambda x: x, [0, 1], 0, 0),
    (lambda x: x, [0, 1], 0.5, 1),
    (lambda x: x, [0, 1], 1, 2),
])
def test_normalize(test_function, test_domain, argument, expected):
    assert normalize(test_function, test_domain)(argument) == expected
