from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA

def check_stationary(selected_price_data):

    prices_1yr = selected_price_data.iloc[-252:]
    prices_3yr = selected_price_data.iloc[-756:]

    tickers = selected_price_data.columns.to_list()
    log_ret_lst_1yr = pd.DataFrame()
    log_ret_lst_3yr =  pd.DataFrame()
    unstationary_1yr = []
    unstationary_3yr = []

    for ticker in tickers:
        log_ret  = np.log(prices_1yr[ticker] / prices_1yr[ticker].shift(1)).dropna()
        result = adfuller(log_ret)
        log_ret_lst_1yr[ticker] = log_ret
        if result[1] > 0.05:
            unstationary_1yr.append(ticker)

    for ticker in tickers:
        log_ret  = np.log(prices_3yr[ticker] / prices_3yr[ticker].shift(1)).dropna()
        result = adfuller(log_ret) 
        log_ret_lst_3yr[ticker] = log_ret
        if result[1] > 0.05:
            unstationary_3yr.append(ticker)

    return log_ret_lst_1yr, log_ret_lst_3yr, unstationary_1yr, unstationary_3yr

# def garch(log_returns):
#     #for c in list(log_returns.columns):
#     garch_model = arch_model(log_returns, mean='constant', vol='Garch', p=1, q=1)
#     garch_fit = garch_model.fit()

#     results = {}

#     results['mu'] = garch_fit.params['mu']
#     results['omega'] = garch_fit.params['omega']
#     results['alpha'] = garch_fit.params['alpha[1]']
#     results['beta'] = garch_fit.params['beta[1]']

#     return results, garch_fit.pvalues

def fit_garch_series(r, scale_pct=True, dist="normal"):
    """
    r : pandas.Series (일간 로그수익률)
    scale_pct : arch 최적화 안정 위해 ×100 할지
    dist : "normal", "t" 등
    """
    if scale_pct:
        r = r * 100           # 0.5% → 0.5  (arch가 선호)

    am  = arch_model(r, mean="Constant", vol="Garch", p=1, q=1, dist=dist)
    res = am.fit(disp="off")

    # ---- 파라미터 ↓
    params = {
        "mu"   : res.params["mu"]          / (100 if scale_pct else 1),  # 다시 소수화
        "omega": res.params["omega"]       / (10000 if scale_pct else 1),
        "alpha": res.params["alpha[1]"],
        "beta" : res.params["beta[1]"],
    }

    # ---- 표준화 잔차 ↓
    eps   = res.resid / (100 if scale_pct else 1)   # 스케일 복구
    sigma = res.conditional_volatility / (100 if scale_pct else 1)
    z     = eps / sigma

    return params, z, res.pvalues
    
def fit_garch_df(ret_df):
    param_list = []
    z_mat      = []

    for col in ret_df.columns:
        p, z, pv = fit_garch_series(ret_df[col])
        param_list.append(p)
        z_mat.append(z.values)

    params_df = pd.DataFrame(param_list, index=ret_df.columns)
    Z         = np.column_stack(z_mat)     # (T, N)
    C         = np.corrcoef(Z, rowvar=False)  # ★ CCC 상관행렬

    return params_df, C, Z

def monte_carlo(T, price, df):

    n_sim = 10000         # 시뮬레이션 횟수
    S0 = price.iloc[-1]  # 현재 종가 기준

    simulated_prices = np.zeros((T, n_sim))
    mu = df['mu']
    omega = df['omega']
    alpha = df['alpha']
    beta = df['beta']
    for i in range(n_sim):
        eps = np.zeros(T)
        sigma2 = np.zeros(T)
        ret = np.zeros(T)
        
        # 초기값
        sigma2[0] = omega / (1 - alpha - beta)  # 장기 분산
        eps[0] = np.sqrt(sigma2[0]) * np.random.normal()
        ret[0] = mu + eps[0]
        
        # GARCH 시뮬루프
        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
            eps[t] = np.sqrt(sigma2[t]) * np.random.normal()
            ret[t] = mu + eps[t]
        
        # 누적 로그수익률 → 가격 경로
        price_path = S0 * np.exp(np.cumsum(ret))
        simulated_prices[:, i] = price_path
    
    return simulated_prices


def correlated_noise(T, n_sim, corr_matrix, clip_value=2.5, seed=42):
    """
    corr  : (N,N) 상관행렬 (양정치여야 함)
    return: (T, n_sim, N)  ← 표준정규·상관 반영
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Cholesky factor L  (corr = L @ L.T)
    L = np.linalg.cholesky(corr_matrix + 1e-12*np.eye(corr_matrix.shape[0]))

    # 2) 샘플 독립 Z
    z = np.random.randn(T, n_sim, corr_matrix.shape[0])

    # 3) 상관 부여:  z_corr = z @ L.T
    z_corr = z @ L.T                        # (...,N)·(N,N) = (...,N)

    # 4) Tail‑clip
    z_corr = np.clip(z_corr, -clip_value, clip_value)

    return z_corr


def monte_carlo_with_correlation(
    T, price, garch_df, corr_matrix, n_sim=1000,
    clip_z=3, vol_cap=0.20, max_ret=0.50
):
    n_assets = corr_matrix.shape[0]
    S0 = price.iloc[-1].astype(float).values

    # 1) 상관 노이즈
    z = correlated_noise(T, n_sim, corr_matrix, clip_value=clip_z)

    sim = np.empty((T, n_sim, n_assets))

    for k, ticker in enumerate(garch_df.index):
        mu_a, ω, α, β = garch_df.loc[ticker, ['mu','omega','alpha','beta']].astype(float)

        mu = mu_a / 252        # ① 연율 → 일간 변환 (필요 시)
        vol_cap2 = vol_cap**2  # ② 분산 캡

        for s in range(n_sim):
            σ2 = np.empty(T)
            ε  = np.empty(T)
            P  = np.empty(T)

            σ2[0] = min(ω / (1-α-β), vol_cap2)
            ε[0]  = np.sqrt(σ2[0]) * z[0, s, k]
            r_prev = np.clip(mu + ε[0], -max_ret, max_ret)
            P[0]  = S0[k]

            for t in range(1, T):
                σ2[t] = ω + α*ε[t-1]**2 + β*σ2[t-1]
                σ2[t] = min(σ2[t], vol_cap2)            # ③ 변동성 캡

                ε[t]  = np.sqrt(σ2[t]) * z[t, s, k]
                r_t   = np.clip(mu + ε[t], -max_ret, max_ret)  # ④ 로그수익률 클립
                P[t]  = P[t-1] * np.exp(r_t)

            sim[:, s, k] = P

    return sim