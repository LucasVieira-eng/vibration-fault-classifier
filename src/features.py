"""Extração básica de características de sinais de vibração."""

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import periodogram
from scipy.stats import kurtosis, skew


FREQUENCY_BANDS_HZ = {
    "0_100hz": (0, 100),
    "100_500hz": (100, 500),
    "500_1500hz": (500, 1500),
    "1500_3000hz": (1500, 3000),
}


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


def extract_spectral_band_features(
    signal,
    sampling_rate_hz: float = 6000.0,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Calcula o RMS espectral em bandas fixas usando periodograma Hann.

    Esta rotina usa apenas os valores finitos de um canal. A frequência de
    amostragem padrão é a documentada para o dataset deste projeto.
    """
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        raise ValueError("O sinal precisa de pelo menos quatro valores finitos.")
    if sampling_rate_hz <= 0:
        raise ValueError("A frequência de amostragem deve ser positiva.")

    frequency, power_density = periodogram(
        x,
        fs=sampling_rate_hz,
        window="hann",
        detrend="constant",
        scaling="density",
    )
    output = {}
    for name, (lower_hz, upper_hz) in (bands or FREQUENCY_BANDS_HZ).items():
        if lower_hz < 0 or upper_hz <= lower_hz or upper_hz > sampling_rate_hz / 2:
            raise ValueError(f"Faixa espectral inválida: {name} ({lower_hz}, {upper_hz}).")
        mask = (frequency >= lower_hz) & (frequency < upper_hz)
        if mask.sum() < 2:
            raise ValueError(f"Resolução insuficiente para calcular a faixa {name}.")
        band_power = trapezoid(power_density[mask], frequency[mask])
        output[f"fft_rms_{name}"] = float(np.sqrt(max(float(band_power), 0.0)))
    return output


