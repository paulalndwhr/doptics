from doptics import functions, single_mirror, solver#, two_mirrors
import pytest


def func_to_test(input_) -> str:
    print(type(input_))
    return 'gommemode'


@pytest.mark.parametrize("test_input,expected", [
    ('lol', 'gommemode'),
    ('sus', 'gommemode'),
    ('cringe', 'gommemode'),
    (None, 'gommemode')
])
def test_func_to_test(test_input, expected):
    assert func_to_test(test_input) is expected


def test_function():
    print(functions.f)
    assert 1 == (3 - 2)
