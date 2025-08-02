import numpy as np
import pandas as pd

# ---------- Rolling Mean (NumPy)
def rolling_mean_numpy(arr, window):
    result = np.convolve(arr, np.ones(window), 'valid') / window
    return result

# ---------- Rolling Variance (NumPy)
def rolling_var_numpy(arr, window):
    means = rolling_mean_numpy(arr, window)
    var = np.convolve(arr**2, np.ones(window), 'valid') / window - means**2
    return var

# ---------- Rolling Covariance (NumPy)
def rolling_cov_numpy(x, y, window):
    mean_x = rolling_mean_numpy(x, window)
    mean_y = rolling_mean_numpy(y, window)
    mean_xy = np.convolve(x * y, np.ones(window), 'valid') / window
    cov = mean_xy - (mean_x * mean_y)
    return cov

# ---------- Rolling Mean (pandas)
def rolling_mean_pandas(series, window):
    return series.rolling(window=window).mean()

# ---------- Rolling Variance (pandas)
def rolling_var_pandas(series, window):
    return series.rolling(window=window).var()

# ---------- Rolling Covariance (pandas)
def rolling_cov_pandas(series1, series2, window):
    return series1.rolling(window=window).cov(series2)

# ---------- EWMA (Exponential Weighted Moving Average using NumPy)
def ewma_numpy(arr, alpha=0.3):
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for t in range(1, len(arr)):
        result[t] = alpha * arr[t] + (1 - alpha) * result[t - 1]
    return result

# ---------- FFT-based Bandpass Filter
def fft_bandpass_filter(signal, low_freq, high_freq, sample_rate):
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1 / sample_rate)
    mask = (freqs > low_freq) & (freqs < high_freq)
    fft_filtered = fft_vals * mask
    return np.fft.ifft(fft_filtered).real

