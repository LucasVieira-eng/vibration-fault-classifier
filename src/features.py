"""Extração básica de características de sinais de vibração."""

import numpy as np
from scipy.stats import kurtosis, skew


def extract_features(signal) -> dict[str, float]:
    """Extrai estatísticas simples de um vetor numérico."""
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("O sinal não contém valores numéricos finitos.")

    rms = float(np.sqrt(np.mean(np.square(x))))
    mean_abs = float(np.mean(np.abs(x)))
    peak_to_peak = float(np.ptp(x))

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": rms,
        "mean_abs": mean_abs,
        "peak": float(np.max(np.abs(x))),
        "peak_to_peak": peak_to_peak,
        "skewness": float(skew(x)) if x.size > 2 else 0.0,
        "kurtosis": float(kurtosis(x)) if x.size > 3 else 0.0,
    }


