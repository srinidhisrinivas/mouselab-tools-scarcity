import pytest

from mouselab.env_utils import get_num_actions


@pytest.mark.parametrize("branching,result", [[[3, 1, 2], 13]])
def test_num_actions(branching, result):
    num_actions = get_num_actions(branching)
    assert num_actions == result
