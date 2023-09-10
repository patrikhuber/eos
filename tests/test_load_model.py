import pytest
import eos

def test_load_model():
    model = eos.morphablemodel.load_model("../share/sfm_shape_3448.bin")
    assert model is not None