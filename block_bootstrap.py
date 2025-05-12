import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import yfinance as yf


def fetch_returns(tickers, start="2020-01-01", end="2025-01-01"):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    returns = np.log(data / data.shift(1)).dropna()
    return returns.T.values  # shape: (num_assets, num_days)


def find_n_star(acf_values, threshold=0.025):
    """
    Finds the largest lag where autocorrelation becomes significant when scanning backwards.
    Returns the first lag (from the end) where |C(n)| > threshold
    """
    for lag in range(len(acf_values) - 1, 0, -1):  # from max_lag to 1
        if abs(acf_values[lag]) > threshold:
            return lag
    return 1  # if no significant correlation found

def estimate_n_star_all_series(data, max_lag=200, threshold=0.025, plot=False):
    """
    data: numpy array (num_series, series_length)
    max_lag: maximum lag for autocorrelation
    threshold: threshold to determine significance
    """
    n_stars = []
    num_series = data.shape[0]

    if plot:
        plt.figure(figsize=(12, 6))

    for i in range(num_series):
        acf_vals = acf(data[i], nlags=max_lag, fft=True)
        n_star = find_n_star(acf_vals, threshold)
        n_stars.append(n_star)

        if plot:
            plt.plot(acf_vals[1:], label=f'Series {i+1}')
    print(n_stars)
    min_n_star = min(n_stars)

    if plot:
        plt.axhline(threshold, color='red', linestyle='--', label='±2.5% threshold')
        plt.axhline(-threshold, color='red', linestyle='--')
        plt.title('Autocorrelation C(n) for all series')
        plt.xlabel('Lag (days)')
        plt.ylabel('C(n)')
        plt.legend()
        plt.grid(True)
        plt.show()

    return n_stars, min_n_star

# Step 2: 블록 부트스트랩 함수
def block_bootstrap(data, block_size, num_blocks, seed=None):
    if seed is not None:
        np.random.seed(seed)

    num_series, series_length = data.shape
    max_start = series_length - block_size
    children = []

    for _ in range(num_blocks):
        start_idx = np.random.randint(0, max_start + 1)
        block = data[:, start_idx:start_idx + block_size]
        children.append(block)

    resampled_data = np.concatenate(children, axis=1)
    return resampled_data

# Step 3: 전체 파이프라인
def run_block_bootstrap_pipeline(data, max_lag=200, threshold=0.025, num_blocks=10, seed=42, plot=True):
    """
    data: numpy array (num_series, series_length)
    Returns: resampled data, common_n_star, list of n* for each series
    """
    # 1. n* 계산
    n_star_list, common_n_star = estimate_n_star_all_series(
        data, max_lag=max_lag, threshold=threshold, plot=plot
    )
    print(f"Block Size (n*): {common_n_star}")

    # 2. 블록 부트스트랩
    resampled_data = block_bootstrap(
        data, block_size=common_n_star, num_blocks=num_blocks, seed=seed
    )
    print(f"Resampled Data: {resampled_data.shape}")

    return resampled_data, common_n_star, n_star_list



if __name__ == "__main__":
    # np.random.seed(0)
    # data = np.cumsum(np.random.normal(0, 0.000001, (32, 1076)), axis=1)  # shape: (32, 1076)

    # # 전체 파이프라인 실행
    # resampled_data, common_n_star, n_star_list = run_block_bootstrap_pipeline(
    #     data, max_lag=200, threshold=0.025, num_blocks=10, seed=2025, plot=True
    # )
    tickers = ["AAPL", "MSFT"]
    return_data = fetch_returns(tickers, start="2020-01-01", end="2023-01-01")