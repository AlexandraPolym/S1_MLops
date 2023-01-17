#  tests/test_model.py
from tests import _PATH_DATA
from data import CorruptMnist
from model import MyAwesomeModel
import numpy as np
import os.path
import pytest
import torch

def test_error_on_wrong_shape():
   with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
      MyAwesomeModel.forward(MyAwesomeModel, torch.randn(1,2,3))