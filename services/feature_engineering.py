"""Feature extraction utilities for inertial signals.

This module exposes a high level :func:`extract_features` function used to
convert a window of 5 seconds of inertial measurements into a vector of 256
features. The implementation combines time and frequency domain metrics and
reduces the dimensionality using a PCA model stored in MLflow.

Example
-------
>>> import numpy as np
>>> from services.feature_engineering import extract_features
>>> window = np.random.randn(500, 6)
>>> features = extract_features(window)
>>> features.shape
(256,)
"""
from __future__ import annotations

import numpy as np
import mlflow
from typing import Iterable

try:
    from sklearn.preprocessing import StandardScaler  # type: ignore
except Exception:  # pragma: no cover - sklearn may not be installed
    StandardScaler = None  # type: ignore

try:
    from sklearn.decomposition import PCA  # type: ignore
except Exception as err:  # pragma: no cover - sklearn may not be installed
    PCA = None  # type: ignore

# ---------------------------------------------------------------------------
# Helper statistics
# ---------------------------------------------------------------------------


def _kurtosis(x: np.ndarray) -> float:
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return np.mean((x - mean) ** 4) / (std ** 4) - 3.0


def _skewness(x: np.ndarray) -> float:
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return np.mean((x - mean) ** 3) / (std ** 3)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def scale_window(
    window: np.ndarray, scaler_uri: str = "models:/signal_scaler/2"
) -> np.ndarray:
    """Scale the raw signal window using a pre-trained ``StandardScaler``.

    The scaler is loaded from MLflow so the normalization parameters can be
    updated without modifying this code.

    Parameters
    ----------
    window : np.ndarray
        Input array of shape ``(500, 6)`` with raw signals.
    scaler_uri : str
        Location of the scaler in MLflow.

    Returns
    -------
    np.ndarray
        Scaled window with the same shape as ``window``.
    """
    if StandardScaler is None:
        raise RuntimeError("scikit-learn is required for window scaling")
    try:
        scaler: StandardScaler = mlflow.sklearn.load_model(scaler_uri)  # type: ignore
    except Exception as e:  # pragma: no cover - depends on external service
        raise RuntimeError(
            "StandardScaler model could not be loaded from MLflow. Register and train it first."
        ) from e
    return scaler.transform(window)

def extract_time_features(window: np.ndarray) -> np.ndarray:
    """Extract time domain features per axis.

    Parameters
    ----------
    window : np.ndarray
        Signal window of shape ``(500, 6)``.

    Returns
    -------
    np.ndarray
        Flattened array with 60 time domain features.
    """
    features: list[float] = []
    for i in range(window.shape[1]):
        x = window[:, i]
        mean = float(np.mean(x))
        std = float(np.std(x))
        rms = float(np.sqrt(np.mean(np.square(x))))
        kurt = float(_kurtosis(x))
        skew = float(_skewness(x))
        max_v = float(np.max(x))
        min_v = float(np.min(x))
        range_v = max_v - min_v
        abs_mean = float(np.mean(np.abs(x)))
        crest = float(np.max(np.abs(x)) / (rms + 1e-8))
        features.extend(
            [
                mean,
                std,
                rms,
                kurt,
                skew,
                max_v,
                min_v,
                range_v,
                abs_mean,
                crest,
            ]
        )
    return np.asarray(features, dtype=float)


BANDS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
SAMPLE_RATE = 100
N_FFT_COEFFS = 30


def extract_freq_features(window: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Extract frequency domain features per axis."""
    n = window.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    features: list[float] = []

    for i in range(window.shape[1]):
        x = window[:, i]
        spectrum = np.fft.rfft(x)
        mag = np.abs(spectrum)
        psd = mag ** 2

        energy = float(psd.sum() / n)
        idx = int(np.argmax(mag[1:]) + 1)
        dom_freq = float(freqs[idx])

        p_norm = psd / (psd.sum() + 1e-12)
        spectral_entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12)))

        band_powers = []
        for low, high in BANDS:
            idx_band = np.where((freqs >= low) & (freqs < high))
            if len(idx_band[0]) == 0:
                band_powers.append(0.0)
            else:
                band_powers.append(float(psd[idx_band].mean()))

        harmonic_freq = 2.0 * dom_freq
        h_idx = int(np.argmin(np.abs(freqs - harmonic_freq)))
        harmonic_amp = float(mag[h_idx])
        dom_amp = float(mag[idx])
        prominence = dom_amp / (harmonic_amp + 1e-8)

        coeffs = mag[:N_FFT_COEFFS].astype(float).tolist()

        features.extend(
            [energy, dom_freq, spectral_entropy, *band_powers, prominence, *coeffs]
        )
    return np.asarray(features, dtype=float)


def extract_envelope_features(window: np.ndarray, smooth: int = 10) -> np.ndarray:
    """Compute envelope energy per axis using rectification and moving average."""
    kernel = np.ones(smooth) / smooth
    feats = []
    for i in range(window.shape[1]):
        x = np.abs(window[:, i])
        env = np.convolve(x, kernel, mode="same")
        energy = float(np.sum(env ** 2) / len(env))
        feats.append(energy)
    return np.asarray(feats, dtype=float)


def reduce_dimensionality(features: np.ndarray, pca_uri: str = "models:/feature_pca/1") -> np.ndarray:
    """Normalize and project features using a pre-trained PCA model."""
    if PCA is None:
        raise RuntimeError("scikit-learn is required for PCA transformation")
    x = features.reshape(1, -1)
    try:
        pca_model: PCA = mlflow.sklearn.load_model(pca_uri)  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PCA model could not be loaded from MLflow. Register and train it first."
        ) from e
    projected = pca_model.transform(x)
    return projected.squeeze()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(window: np.ndarray) -> np.ndarray:
    """Generate a 256-length feature vector from a ``(500, 6)`` window.

    Parameters
    ----------
    window : np.ndarray
        Input array containing inertial signals for 5 seconds with 6 channels.

    Returns
    -------
    np.ndarray
        Vector of 256 features obtained after PCA projection.
    """
    if window.shape != (500, 6):
        raise ValueError("window must have shape (500, 6)", window.shape)

    # Apply static normalization to the input window
    window = scale_window(window)

    time_f = extract_time_features(window)
    freq_f = extract_freq_features(window)
    env_f = extract_envelope_features(window)
    full_features = np.concatenate([time_f, freq_f, env_f])
    return reduce_dimensionality(full_features)