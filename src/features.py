"""Extração básica de características de sinais de vibração."""

import numpy as np
import pandas as pd
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


def extract_dataframe_features(
    df: pd.DataFrame, signal_columns: list[str]
) -> pd.DataFrame:
    """Extrai uma linha de features por coluna de sinal."""
    rows = []
    for column in signal_columns:
        if column not in df.columns:
            raise KeyError(f"Coluna inexistente: {column}")
        row = extract_features(df[column].to_numpy())
        row["signal_column"] = column
        rows.append(row)
    return pd.DataFrame(rows)
