#test for the fixed-point refernence implementation

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lr_inference import (
    QFormat, load_model, quantize, dequantize, quantize_model,
    classify_fixed, classify_batch, required_acc_bits, consensus,
)

ROOT = Path(__file__).resolve().parents[1]

#The chosen deployment format
X_FMT = QFormat(16,8)
W_FMT = QFormat(16,1)
ACC_WIDTH = 32

@pytest.fixture(scope= "module")
def model():
    return load_model(ROOT / "models" / "lr_weights.npz")

@pytest.fixture(scope="module")
def vectors():
    return np.load(ROOT / "models" / "holdout_vectors.npz")


@pytest.fixture(scope="module")
def quantized(model, vectors):
    w_q, bias_q, _ = quantize_model(model, W_FMT, X_FMT)
    x_q, _ = quantize(vectors["X_std"], X_FMT)
    bits, acc = classify_batch(x_q, w_q, bias_q)
    return x_q, w_q, bias_q, bits, acc


# ----------- Q format ----------------

def test_qformat_fields():
    f = QFormat(16, 5)
    assert f.frac_bits == 11
    assert f.scale == 2048
    assert f.max_int == 32767
    assert f.min_int == -32768
    assert f.max_float == pytest.approx(15.99951, abs=1e-4)


def test_quantize_roundtrip():
    x = np.array([0.0, 1.5, -3.25, 7.125])
    q, sat = quantize(x, QFormat(16, 5))
    assert sat == 0
    assert np.allclose(dequantize(q, QFormat(16, 5)), x)


def test_quantize_saturates_and_counts():
    q, sat = quantize(np.array([100.0, -100.0, 1.0]), QFormat(16, 5))
    assert sat == 2
    assert q[0] == QFormat(16, 5).max_int
    assert q[1] == QFormat(16, 5).min_int


def test_quantize_returns_integers():
    q, _ = quantize(np.array([1.234]), QFormat(16, 5))
    assert np.issubdtype(q.dtype, np.integer), "reference must be integer-only"


# -------weights -----------

def test_chosen_format_preserves_every_weight(model):
    w_q, _, sat = quantize_model(model, W_FMT, X_FMT)
    assert sat == 0
    assert np.all(w_q != 0), "a weight quantized to zero, feature silently dropped"


def test_q88_destroys_a_weight(model):
    """Documents why the original Q8.8 specification is unusable."""
    w_q, _, _ = quantize_model(model, QFormat(16, 8), QFormat(16, 8))
    assert np.any(w_q == 0), "expected Q8.8 to zero at least one coefficient"


# ----------- inference ----------------

def test_scalar_matches_batch(quantized):
    x_q, w_q, bias_q, bits, _ = quantized
    rng = np.random.default_rng(42)
    for i in rng.choice(len(x_q), 200, replace=False):
        assert classify_fixed(x_q[i], w_q, bias_q) == int(bits[i])


def test_accumulator_fits_declared_width(quantized):
    *_, acc = quantized
    need = required_acc_bits(acc)
    assert need <= ACC_WIDTH, f"accumulator needs {need} bits, declared {ACC_WIDTH}"


def test_fixed_point_matches_float_model(quantized, vectors):
    *_, bits, _ = quantized
    match = (bits == vectors["relay_bit"]).mean()
    assert match >= 0.998, f"relay bit agreement only {match:.6f}"



# ----------- consensus ----------------

def test_consensus_is_or_rule():
    assert consensus([0, 0, 0, 0]) == 0
    assert consensus([1, 0, 0, 0]) == 1
    assert consensus([0, 0, 1, 0]) == 1
    assert consensus([1, 1, 1, 1]) == 1


def test_consensus_threshold_is_configurable():
    assert consensus([1, 1, 0, 0], threshold=2) == 1
    assert consensus([1, 0, 0, 0], threshold=2) == 0


def test_consensus_reproduces_saved_snapshots(quantized, vectors, model):
    *_, bits, _ = quantized
    order = np.lexsort((vectors["relay_id"], vectors["orig_row"]))
    assert np.array_equal(np.unique(vectors["orig_row"]), vectors["snapshot_id"])

    grouped = bits[order].reshape(-1, 4)
    snap_pred = (grouped.sum(axis=1) >= model.vote_threshold).astype(np.int8)

    match = (snap_pred == vectors["snapshot_or"]).mean()
    assert match >= 0.998, f"relay bit agreement only {match:.6f}"
