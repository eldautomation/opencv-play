import pytest
from autocollimator.target_utils import pct_list_to_int_list


def test_pct_list_to_int_list_basic():
    data = [.102, .217, .227 , .305]
    scale = 100
    result = pct_list_to_int_list(data,scale)
    assert result == [10, 20, 22, 30]

def test_pct_list_to_int_list_empty():
    scale = 100
    assert pct_list_to_int_list([],scale) == []


def test_pct_list_to_int_list_invalid_type():
    with pytest.raises(TypeError):
        pct_list_to_int_list([10, "bad", 30])


def test_pct_list_to_int_list_none():
    with pytest.raises(TypeError):
        pct_list_to_int_list(None)
