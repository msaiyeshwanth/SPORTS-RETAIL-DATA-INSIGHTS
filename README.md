# SPORTS-RETAIL-DATA-INSIGHTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from statsmodels.graphics.tsaplots import plot_acf

# Example data (replace this with your real dataset)
df = pd.DataFrame({
    'Time': pd.date_range(start='1/1/2020', periods=200, freq='H'),
    'East_Air_Flow': np.random.normal(loc=55, scale=2, size=200)  # Simulated data
})

# 1. Fast Fourier Transform (FFT) to detect cycles
def detect_cycles_fft(airflow_series, sampling_rate=1):
    # Apply FFT
    fft_values = fft(airflow_series)
    
    # Calculate the frequencies
    n = len(airflow_series)
    freqs = np.fft.fftfreq(n, d=sampling_rate)
    
    # Take the magnitude of the FFT and consider only the positive frequencies
    magnitudes = np.abs(fft_values[:n//2])
    positive_freqs = freqs[:n//2]
    
    # Plot the FFT results
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs, magnitudes)
    plt.title('FFT of East Air Flow')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()

# 2. Autocorrelation to detect cycles or repeating patterns
def detect_cycles_autocorrelation(airflow_series, lags=50):
    # Plot the autocorrelation function
    plt.figure(figsize=(10, 6))
    plot_acf(airflow_series, lags=lags)
    plt.title('Autocorrelation of East Air Flow')
    plt.show()

# 3. Rolling standard deviation to quantify variation
def calculate_rolling_std(airflow_series, window_size=10):
    # Calculate the rolling standard deviation
    rolling_std = airflow_series.rolling(window=window_size).std()
    
    # Plot the original data and the rolling std
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time'], airflow_series, label='East Air Flow', color='blue', alpha=0.5)
    plt.plot(df['Time'], rolling_std, label=f'Rolling Std (window={window_size})', color='red', alpha=0.8)
    plt.title(f'East Air Flow and Rolling Standard Deviation (window={window_size})')
    plt.xlabel('Time')
    plt.ylabel('Air Flow / Rolling Std')
    plt.legend()
    plt.show()
    
    return rolling_std

# Usage
detect_cycles_fft(df['East_Air_Flow'])
detect_cycles_autocorrelation(df['East_Air_Flow'])
rolling_std = calculate_rolling_std(df['East_Air_Flow'], window_size=10)
