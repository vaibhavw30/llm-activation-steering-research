# tests/test_sae_load.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from sae_load import pick_l0_path, decoder_unit


FILES = [
    "layer_11/width_16k/average_l0_21/params.npz",
    "layer_11/width_16k/average_l0_68/params.npz",
    "layer_11/width_16k/average_l0_176/params.npz",
    "layer_11/width_65k/average_l0_72/params.npz",
    "layer_20/width_16k/average_l0_71/params.npz",
    "README.md",
]


def test_picks_the_l0_nearest_the_target_for_the_requested_layer_and_width():
    assert pick_l0_path(FILES, 11) == "layer_11/width_16k/average_l0_68/params.npz"
    assert pick_l0_path(FILES, 20) == "layer_20/width_16k/average_l0_71/params.npz"


def test_respects_the_width_argument():
    assert pick_l0_path(FILES, 11, width="65k").startswith("layer_11/width_65k/")


def test_raises_when_no_file_matches():
    with pytest.raises(SystemExit):
        pick_l0_path(FILES, 25)


def test_decoder_unit_normalizes_rows():
    D = decoder_unit({"W_dec": np.array([[3.0, 4.0], [0.0, 2.0]])})
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0)
    assert np.allclose(D[0], [0.6, 0.8])
