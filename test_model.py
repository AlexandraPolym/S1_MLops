from tests import _PATH_DATA
from model import MyAwesomeModel
import pytest


def test_error_on_wrong_shape():
#    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#       model(torch.randn(1,2,3))
    assert 0