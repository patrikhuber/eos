import pytest
import eos
import numpy as np
from pathlib import Path

def test_load_model():
    model = eos.morphablemodel.load_model(str(Path(__file__).resolve().parent.parent / 'share' / 'sfm_shape_3448.bin'))
    assert model is not None