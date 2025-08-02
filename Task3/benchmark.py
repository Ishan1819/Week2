import pandas as pd
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from timeseries import (
    rolling_mean_numpy, rolling_mean_pandas,
    rolling_var_numpy, rolling_var_pandas,
    rolling_cov_numpy, rolling_cov_pandas
)

def benchmark_all_metrics():
    window = 50
    df = pd.read_csv("synthetic_data.csv")
    records = []

    for sensor in ['sensor_1', 'sensor_2', 'sensor_3']:
        series = df[sensor]
        mem_numpy = sys.getsizeof(series.values)
        mem_pandas = sys.getsizeof(series)

        # Rolling Mean
        start = time.time()
        _ = rolling_mean_numpy(series.values, window)
        time_numpy = time.time() - start

        start = time.time()
        _ = rolling_mean_pandas(series, window)
        time_pandas = time.time() - start

        faster = "NumPy" if time_numpy < time_pandas else "pandas"

        records.append({"Sensor": sensor, "Metric": "Rolling Mean", "Method": "NumPy", "Time (s)": round(time_numpy, 6), "Memory (bytes)": mem_numpy})
        records.append({"Sensor": sensor, "Metric": "Rolling Mean", "Method": "pandas", "Time (s)": round(time_pandas, 6), "Memory (bytes)": mem_pandas})
        print(f"✅ {sensor} Rolling Mean: {faster} is faster")

        # Rolling Variance
        start = time.time()
        _ = rolling_var_numpy(series.values, window)
        time_numpy = time.time() - start

        start = time.time()
        _ = rolling_var_pandas(series, window)
        time_pandas = time.time() - start

        faster = "NumPy" if time_numpy < time_pandas else "pandas"

        records.append({"Sensor": sensor, "Metric": "Rolling Variance", "Method": "NumPy", "Time (s)": round(time_numpy, 6), "Memory (bytes)": mem_numpy})
        records.append({"Sensor": sensor, "Metric": "Rolling Variance", "Method": "pandas", "Time (s)": round(time_pandas, 6), "Memory (bytes)": mem_pandas})
        print(f"✅ {sensor} Rolling Variance: {faster} is faster")

    # Rolling Covariance: sensor_1 vs sensor_2
    series1 = df['sensor_1']
    series2 = df['sensor_2']
    mem_numpy = sys.getsizeof(series1.values) + sys.getsizeof(series2.values)
    mem_pandas = sys.getsizeof(series1) + sys.getsizeof(series2)

    start = time.time()
    _ = rolling_cov_numpy(series1.values, series2.values, window)
    time_numpy = time.time() - start

    start = time.time()
    _ = rolling_cov_pandas(series1, series2, window)
    time_pandas = time.time() - start

    faster = "NumPy" if time_numpy < time_pandas else "pandas"
    print(f"✅ Covariance sensor_1 vs sensor_2: {faster} is faster")

    records.append({"Sensor": "sensor_1 vs sensor_2", "Metric": "Rolling Covariance", "Method": "NumPy", "Time (s)": round(time_numpy, 6), "Memory (bytes)": mem_numpy})
    records.append({"Sensor": "sensor_1 vs sensor_2", "Metric": "Rolling Covariance", "Method": "pandas", "Time (s)": round(time_pandas, 6), "Memory (bytes)": mem_pandas})

    # Save and plot
    results_df = pd.DataFrame(records)
    results_df.to_csv("benchmark_results.csv", index=False)
    print("\n🔍 Benchmark completed. Results saved to benchmark_results.csv")

    # 📈 Plot time chart
    pivot_df = results_df.pivot_table(index=['Sensor', 'Metric'], columns='Method', values='Time (s)')
    pivot_df.plot(kind='barh', figsize=(10, 8), title='Execution Time: NumPy vs pandas')
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("benchmark_chart.png")
    plt.show()

if __name__ == "__main__":
    benchmark_all_metrics()
