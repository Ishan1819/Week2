**Task3: High-Performance Time Series Transformation with NumPy & pandas**

____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
**Overview**
This project implements efficient, vectorized time-series transformations for large-scale data (e.g., >1 million rows). It leverages both NumPy and pandas to compute rolling statistics, exponentially weighted moving averages (EWMA), and FFT-based spectral filtering. Performance is benchmarked across methods, with recommendations for the fastest and most memory-efficient approaches.

____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
**Features & Techniques**

**Implemented Functionalities**

Rolling Window Statistics

Mean and variance using both NumPy and pandas

NumPy accelerated via vectorization and convolution

Exponentially Weighted Moving Averages (EWMA)

Efficient NumPy implementation for smoothed trend analysis

Rolling Covariance

Compare multiple sensor streams over moving windows

FFT-Based Spectral Analysis

Band-pass filter using NumPy’s FFT routines for frequency-domain insights

Performance Benchmarks

Runtime and memory profiling for all metrics

Visual comparison using horizontal bar charts

__________________________________________________________________________________________________________________________________
**Requirements**
Install required packages via pip:

pip install numpy pandas matplotlib

1. Generate Synthetic Dataset

python synthetic_data_gen.py

Creates synthetic_data.csv with >1M rows of simulated sensor data.

2. Run Benchmarks

python benchmark.py
Outputs:

benchmark_results.csv (time & memory data)

benchmark_chart.png (bar chart of runtimes)


__________________________________________________________________________________________________________________________________
**Key Functions**

Function	Description

rolling_mean_numpy	Vectorized moving average via np.convolve

rolling_var_numpy	Rolling variance using mean of squares

rolling_cov_numpy	Rolling covariance of two series

rolling_mean_pandas	Built-in .rolling().mean()

rolling_var_pandas	Built-in .rolling().var()

rolling_cov_pandas	Built-in .rolling().cov()

ewma_numpy	Manual implementation of EWMA

fft_bandpass_filter	FFT-based frequency filter


__________________________________________________________________________________________________________________________________
**Benchmark Output**

Metrics Collected

Execution Time (seconds)

Memory Usage (bytes)

Across methods and sensors: sensor_1, sensor_2, sensor_3

__________________________________________________________________________________________________________________________________
**Performance Insights**

NumPy is faster for rolling mean and variance on large arrays due to optimized convolution.

pandas offers better readability and is efficient for small to medium-sized datasets.

NumPy excels in memory efficiency and raw speed when optimized (e.g., with stride_tricks, not shown here).

Automatic selection logic can be implemented in future to switch methods based on data size.
