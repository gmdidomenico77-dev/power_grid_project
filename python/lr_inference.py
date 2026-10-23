from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass(frozen=True)
class QFormat:
    #Signed fixed-point format
    #int_bits includes the sign bit
    total_bits: int
    int_bits: int

    @property
    def frac_bits(self) -> int:
        return self.total_bits - self.int_bits

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def max_int(self) -> int:
        return (1 << (self.total_bits - 1)) - 1

    @property
    def min_int(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def max_float(self) -> float:
        return self.max_int / self.scale

    @property
    def min_float(self) -> float:
        return self.min_int / self.scale

    @property
    def resolution(self) -> float:
        return 1 / self.scale

    def __str__(self) -> str:
        return(f"ap_fixed<{self.total_bits}, {self.int_bits}> "
               f"range [{self.min_float:.4f}, {self.max_float:.4f}] "
               f"res {self.resolution:.3e}")


def quantize(x, fmt: QFormat) -> np.ndarray:
    #Float to integeter with saturating clip. Returns (ints, n_saturated).
    scaled = np.rint(np.asarray(x, dtype=np.float64) * fmt.scale)
    saturated = int(np.sum((scaled > fmt.max_int) | (scaled < fmt.min_int)))
    clipped = np.clip(scaled, fmt.min_int, fmt.max_int).astype(np.int64)
    return clipped, saturated

def dequantize(x, fmt: QFormat) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) / fmt.scale

# ----- model -------
@dataclass
class Model:
    feature_names: list
    coef: np.ndarray
    intercept: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    paz_cap: float
    vote_threshold: int

def load_model(path="models/lr_weights.npz") -> Model:
    d = np.load(Path(path))
    return Model(
        feature_names=[str(s) for s in d["feature_names"]],
        coef=d["coef"].astype(np.float64),
        intercept =float(d["intercept"]),
        scaler_mean  =d["scaler_mean"].astype(np.float64),
        scaler_scale = d["scaler_scale"].astype(np.float64),
        paz_cap=float(d["paz_cap"]),
        vote_threshold=int(d["vote_threshold"])
    )

#------host-side preprocessing -------

def preprocess(raw, model: Model) -> np.ndarray:
    #Raw 8 features to 9 standardized inputs, matching training pipeline

    #Columns must be in model.feature_names order minus PA:Z__open
    #PA:Z may be +inf, which is what marks an open breaker

    x = np.array(raw, dtype = np.float64, copy = True)
    if x.ndim  ==1:
        x = x[None, :]
    assert x.shape[1] ==len(model.feature_names) -1, "expected 8 raw features"

    paz = model.feature_names.index("PA:Z")
    open_flag = np.isposinf(x[:, paz]).astype(np.float64) #flag before we cap
    x[:, paz] = np.minimum(x[:, paz], model.paz_cap) #cap PA:Z to avoid inf

    full = np.concatenate([x, open_flag[:, None]], axis=1)
    return (full - model.scaler_mean) / model.scaler_scale

#---- silicon-side inference ----

def quantize_model(model: Model, w_fmt: QFormat, x_fmt: QFormat):
    #quantize weights and bias. Bias is caled to the product scale
    w_q, w_sat = quantize(model.coef, w_fmt)
    bias_q = int(np.rint(model.intercept * w_fmt.scale * x_fmt.scale))
    return w_q, bias_q, w_sat

def classify_fixed(x_q, w_q, bias_q) -> int:
    #Integer-only inference for one relay. Returns 1 (Attack) or 0 (Natural)
    acc = int(bias_q)
    for xi, wi in zip(x_q, w_q):
        acc+= int(xi) * int(wi)
    return 1 if acc >= 0 else 0

def classify_batch(x_q, w_q, bias_q):
    #Vectorized over rows; returns (bits, accumulators)
    x = np.asarray(x_q, dtype =np.int64)
    w = np.asarray(w_q, dtype = np.int64)
    acc = x @ w + bias_q
    return (acc >= 0).astype(np.int8), acc

def required_acc_bits(acc) -> int:
    """Minimum signed width that holds these accumulator values."""
    return int(np.abs(np.asarray(acc, dtype=np.int64)).max()).bit_length() + 1

# ------ consensus -------

def consensus(bits, threshold: int =1) -> int:
    #Combine relay bits
    #threshold = 1 is the OR rule
    return int(int(np.sum(bits))>= threshold)