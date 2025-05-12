import numpy as np
import pandas as pd
import yfinance as yf
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt

def plot_efficient_frontier(results_df_1, risk):
    """
    Efficient Frontier를 시각화하는 함수
    """
    # 데이터 준비
    gamma = results_df_1["gamma"]
    risk = results_df_1[risk]
    reward = results_df_1["ObjVal"]
    sharpe_ratio = results_df_1["Reward/Risk"]

    # 그래프 설정
    plt.figure(figsize=(10, 6))
    
    # Efficient Frontier Plot
    plt.plot(risk, reward, marker='o', linestyle='-', color='blue', label='Efficient Frontier')
    
    # 최적 포인트 (샤프 비율 최대)
    max_sharpe_idx = sharpe_ratio.idxmax()
    plt.scatter(risk[max_sharpe_idx], reward[max_sharpe_idx], color='red', s=100, marker='*', label='Max Sharpe Ratio')

    # 각 포인트에 gamma 값 표시
    for i in range(len(risk)):
        plt.annotate(f"{gamma[i]:.2f}", (risk[i], reward[i]), textcoords="offset points", xytext=(0, 5), ha='center')

    # 그래프 제목과 레이블 설정
    plt.title("Efficient Frontier")
    plt.xlabel("Risk")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_reward_avgdd(df_list, alpha_lst):
    """
    Reward-MaxDD 그래프를 그리는 함수
    - df_list: 각 제약 수준별 결과 DataFrame 리스트
    - alpha_lst: 각 DataFrame의 알파 리스트
    """
    plt.figure(figsize=(8, 6))
    
    markers = ['d', 's', '^', 'o']  # 마커 모양
    colors = ['red', 'green', 'cyan', 'blue']  # 색상

    for i, (df, alpha) in enumerate(zip(df_list, alpha_lst)):
        # x: MaxDD, y: Expected Return
        risk = df.loc['avg_dd'].values 
        reward = df.loc['ObjVal'].values
        label = f"{int((1 - alpha) * 100)}% CDaR"
        plt.plot(risk, reward, marker=markers[i], linestyle='-', color=colors[i], label=label)
        
        # 각 점에 gamma 값 표시
        # for j in range(len(risk)):
        #     plt.annotate(f"{df.loc['gamma'].values[j]:.2f}", 
        #                  (risk[j], reward[j]), 
        #                  textcoords="offset points", xytext=(0, 5), ha='center')

    # 그래프 레이아웃 설정
    plt.title("Reward-MaxDD")
    plt.xlabel("AvgDD (A(x))")
    plt.ylabel("Expected Return (R(x))")
    plt.legend(title="CDaR Constraint")
    plt.grid(True)
    plt.show()

def plot_reward_maxdd(df_list, alpha_lst):
    """
    Reward-MaxDD 그래프를 그리는 함수
    - df_list: 각 제약 수준별 결과 DataFrame 리스트
    - alpha_lst: 각 DataFrame의 알파 리스트
    """
    plt.figure(figsize=(8, 6))
    
    markers = ['d', 's', '^', 'o']  # 마커 모양
    colors = ['red', 'green', 'cyan', 'blue']  # 색상

    for i, (df, alpha) in enumerate(zip(df_list, alpha_lst)):
        # x: MaxDD, y: Expected Return
        risk = df.loc['max_dd'].values if 'max_dd' in df.index else df.loc['cdar'].values
        reward = df.loc['ObjVal'].values
        label = f"{int((1 - alpha) * 100)}% CDaR"
        plt.plot(risk, reward, marker=markers[i], linestyle='-', color=colors[i], label=label)
        
        
        for j in range(len(risk)):
            plt.annotate(f"{df.loc['gamma'].values[j]:.2f}", 
                         (risk[j], reward[j]), 
                         textcoords="offset points", xytext=(0, 5), ha='center')

    # 그래프 레이아웃 설정
    plt.title("Reward-MaxDD")
    plt.xlabel("MaxDD (M(x))")
    plt.ylabel("Expected Return (R(x))")
    plt.legend(title="CDaR Constraint")
    plt.grid(True)
    plt.show()

def price_data(selected_price_data):

    prices_1yr = selected_price_data.iloc[-253:]
    prices_3yr = selected_price_data.iloc[-757:]
    returns_1yr = prices_1yr.pct_change().dropna()
    cum_returns_1yr = returns_1yr.cumsum()
    returns_3yr = prices_3yr.pct_change().dropna()
    cum_returns_3yr = returns_3yr.cumsum()

    return cum_returns_1yr, cum_returns_3yr

def maxdd_model(y, gamma, x_min, x_max, leverage = None):
    """
    y : (T×N) ‑ DataFrame, t‑시점까지 **누적** 수익률
    gamma : 허용 최대 드로우다운 (y 단위와 같아야 함)
    """
    T, N = y.shape
    m = Model("MaxRet_MaxDD")

    # 1) 의사결정변수
    x = m.addVars(N, lb=x_min, ub=x_max, vtype=GRB.CONTINUOUS, name="x")   # 포트폴리오 비중
    u = m.addVars(T, vtype=GRB.CONTINUOUS, name="u")                       # running max
    max_dd = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="max_dd")

    # 2) 누적수익식 w_t 와 제약
    for t in range(T):
        w_t = quicksum(x[i] * y.iloc[t, i] for i in range(N))              # (선형식)

        m.addConstr(u[t] >= w_t,           name=f"u_ge_w_{t}")             # u_t ≥ w_t
        if t > 0:
            m.addConstr(u[t] >= u[t-1],    name=f"u_ge_prev_{t}")          # u_t ≥ u_{t-1}

        m.addConstr(u[t] - w_t <= gamma,   name=f"dd_bound_{t}")           # DD ≤ γ
    m.addConstr(max_dd <= gamma, name="maxdd_bound")
    # 3) 예산(레버리지 금지)
    if leverage == None:
        m.addConstr(quicksum(x[i] for i in range(N)) == 1, name="budget")

    # 4) 목적함수 – 연환산 수익률 최대화
    mu = (1 + y.iloc[-1]) ** (252 / T) - 1        # 벡터(길이 N)
    m.setObjective(quicksum(x[i] * mu[i] for i in range(N)), GRB.MAXIMIZE)

    m.update()
    return m, x, u, max_dd 

# def avgdd_model(y, gamma, x_min, x_max, leverage = None):
#     """
#     y : (T×N) pandas.DataFrame - 각 자산의 시점별 누적 수익률
#     gamma : 허용 가능한 평균 드로우다운 (예: 0.15 = 15%)
#     """

#     T, N = y.shape
#     model = Model("Maximize_Return_with_AvgDD_Constraint")

#     # 1. 변수 정의
#     x = model.addVars(N, lb=x_min, ub=x_max, vtype=GRB.CONTINUOUS, name="x")  # 포트폴리오 비중
#     u = model.addVars(T, vtype=GRB.CONTINUOUS, name="u")                      # running max at t
#     model.addConstr(u[0] == 0, name="u_initial")
#     # 2. drawdown 제약조건 및 정의
#     avgdd_expr = 0
#     for t in range(T):
#         w_t = quicksum(x[i] * y.iloc[t, i] for i in range(N))                # w_t = xᵗ y_t

#         model.addConstr(u[t] >= w_t, name=f"u_ge_w_{t}")                     # u_t ≥ w_t
#         if t > 0:
#             model.addConstr(u[t] >= u[t - 1], name=f"u_monotone_{t}")       # u_t ≥ u_{t-1}

#         avgdd_expr += u[t] - w_t                                            # D_t = u_t - w_t

#     # 3. 평균 드로우다운 제약 (논문 기준)
#     model.addConstr((1 / T) * avgdd_expr <= gamma, name="avgdd_bound")

#     # 4. 예산 제약
#     if leverage == None:
#         model.addConstr(quicksum(x[i] for i in range(N)) == 1, name="budget")

#     # 5. 목적함수: annualized return 최대화
#     mu = (1 + y.iloc[-1]) ** (252 / T) - 1   # 연환산 수익률
#     model.setObjective(quicksum(x[i] * mu[i] for i in range(N)), GRB.MAXIMIZE)

#     model.update()
#     return model, x, u

def avgdd_model(y, gamma, x_min, x_max, leverage=None):
    """
    y : (T,N) 누적수익률 DataFrame
    gamma : 허용 AvDD (ex 0.015 = 1.5%)
    """
    T, N = y.shape
    m = Model("MaxRet_AvDD")

    # ── 결정변수 ────────────────
    x = m.addVars(N, lb=x_min, ub=x_max, name="x")
    u = m.addVars(T, lb=0, name="u")          # running max
    dd = m.addVars(T, lb=0, name="dd")        # drawdown = u - w

    # ── 누적 wealth & drawdown ──
    for t in range(T):
        w_t = quicksum(x[i] * y.iloc[t,i] for i in range(N))

        m.addConstr(u[t] >= w_t)                          # u ≥ w
        if t > 0:
            m.addConstr(u[t] >= u[t-1])                   # non‑dec

        m.addConstr(dd[t] == u[t] - w_t)                  # DD_t

    # AvDD 제약  (평균)
    m.addConstr((1/T) * quicksum(dd[t] for t in range(T)) <= gamma,
                name="avdd_bound")

    # 예산
    if leverage is None:
        m.addConstr(quicksum(x) == 1, name="budget")

    # 목적: 연환산 기대수익
    mu = (1 + y.iloc[-1]) ** (252 / T) - 1
    m.setObjective(quicksum(x[i] * mu[i] for i in range(N)), GRB.MAXIMIZE)

    m.update()
    return m, x, u, dd

def cdar_model(y, gamma, alpha, x_min, x_max, leverage = None):
    """
    y     : (T×N) DataFrame, t‑시점까지 누적수익률
    gamma : 허용 최대 CDaR
    alpha : 신뢰수준 (ex. 0.95)
    """
    T, N = y.shape
    m = Model("MaxRet_CDaR_MaxDD")

    # 1) 의사결정변수
    x  = m.addVars(N, lb=x_min, ub=x_max, vtype=GRB.CONTINUOUS, name="x")   # 비중
    u  = m.addVars(T, vtype=GRB.CONTINUOUS, name="u")                       # running max
    z0 = m.addVar(vtype=GRB.CONTINUOUS, name="z0")                          # 스칼라 z
    z  = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="z")               # 벡터 z_k
    max_dd = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="max_dd")          # Max Drawdown

    # 2) running max, drawdown, 초과분 z_k
    for t in range(T):
        w_t = quicksum(x[i] * y.iloc[t, i] for i in range(N))               # 누적수익 w_t

        # Running Max 계산
        m.addConstr(u[t] >= w_t,          name=f"u_ge_w_{t}")
        if t > 0:
            m.addConstr(u[t] >= u[t-1],   name=f"u_monotone_{t}")

        # Drawdown 계산
        drawdown = u[t] - w_t
        m.addConstr(z[t] >= drawdown - z0, name=f"z_excess_{t}")            # z_k ≥ DD - z0

        # MaxDD 계산 제약
        m.addConstr(max_dd >= drawdown, name=f"maxdd_ge_dd_{t}")

    # u_0 = 0
    m.addConstr(u[0] == 0, name="u0_zero")

    # 3) CDaR 제약 :  z0 + 1/((1-α)T) Σ z_k ≤ γ
    cdar_factor = 1 / ((1 - alpha) * T)
    m.addConstr(z0 + cdar_factor * quicksum(z[t] for t in range(T)) <= gamma,
                name="cdar_bound")

    # 4) 예산 (레버리지 금지)
    if leverage is None:
        m.addConstr(quicksum(x[i] for i in range(N)) == 1, name="budget")

    # 5) 목적함수 – 연환산 수익률 최대화
    mu = (1 + y.iloc[-1]) ** (252 / T) - 1
    m.setObjective(quicksum(x[i] * mu[i] for i in range(N)), GRB.MAXIMIZE)

    m.update()
    return m, x, u, z0, z, max_dd
def cdar_model_compare_avgdd(y, gamma, alpha, x_min, x_max,
               leverage=None):

    """
    mode = "cdar" :  CDaR + MaxDD 제약  (논문 식 18)
    mode = "avdd" :  AvDD  (=평균 DD) 제약만 걸고 MaxDD·CDaR은 post calc
    """

    T, N = y.shape
    m = Model(f"MaxRet_{mode.upper()}")

    # ── 변수 ───────────────────────
    x  = m.addVars(N, lb=x_min, ub=x_max, name="x")
    u  = m.addVars(T, lb=0, name="u")          # running max
    dd = m.addVars(T, lb=0, name="dd")         # drawdown
    maxdd = m.addVar(lb=0, name="maxdd")       # MaxDD

    # CDaR 모드 전용 변수
    if mode == "cdar":
        z0 = m.addVar(lb=0, ub=gamma, name="z0")
        z  = m.addVars(T, lb=0, name="z")

    # ── 수익·DD 관계 ───────────────
    for t in range(T):
        w_t = gp.quicksum(x[i] * y.iloc[t, i] for i in range(N))

        m.addConstr(u[t] >= w_t)
        if t > 0:
            m.addConstr(u[t] >= u[t-1])

        m.addConstr(dd[t] == u[t] - w_t)           # DD_t
        m.addConstr(maxdd >= dd[t])

        if mode == "cdar":
            m.addConstr(z[t] >= dd[t] - z0)        # z_k ≥ DD - z0

    # ── 리스크 제약 ────────────────
    if mode == "avdd":
        m.addConstr((1/T) * gp.quicksum(dd) <= gamma, name="avdd_bound")

    else:  # CDaR
        factor = 1/((1-alpha)*T)
        m.addConstr(z0 + factor*gp.quicksum(z) <= gamma, name="cdar_bound")
        m.addConstr(maxdd <= gamma, name="maxdd_cap")  # (선택) MaxDD 바인딩

    # ── 예산 ───────────────────────
    if leverage is None:
        m.addConstr(gp.quicksum(x) == 1)

    # ── 목적 : 연환산 기대수익 최대 ──
    mu = (1 + y.iloc[-1])**(252/T) - 1
    m.setObjective(gp.quicksum(x[i] * mu[i] for i in range(N)),
                   GRB.MAXIMIZE)

    m.optimize()

    # ── 결과 정리 ──────────────────
    w_opt = np.array([x[i].X for i in range(N)])
    path  = y.values @ w_opt
    run   = np.maximum.accumulate(path)
    dd_np = run - path

    result = {
        "R": m.ObjVal,
        "MaxDD": dd_np.max(),
        "AvDD": dd_np.mean(),
        "weights": w_opt,
        
    }
    result["CDaR"] = z0.X + factor * sum(z[t].X for t in range(T))
    # if mode == "cdar":
    #     result["CDaR"] = z0.X + factor * sum(z[t].X for t in range(T))

    return m, x, u, z0, z, result["AvDD"], result["CDaR"]

def cdar_model_compare_maxdd(y, gamma, alpha, x_min, x_max, leverage = None):
    """
    y     : (T×N) DataFrame, t‑시점까지 누적수익률
    gamma : 허용 최대 CDaR
    alpha : 신뢰수준 (ex. 0.95)
    """
    T, N = y.shape
    m = Model("MaxRet_CDaR_MaxDD")

    # 1) 의사결정변수
    x  = m.addVars(N, lb=x_min, ub=x_max, vtype=GRB.CONTINUOUS, name="x")   # 비중
    u  = m.addVars(T, vtype=GRB.CONTINUOUS, name="u")                       # running max
    z0 = m.addVar(vtype=GRB.CONTINUOUS, name="z0")                          # 스칼라 z
    z  = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="z")               # 벡터 z_k
    max_dd = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="max_dd")          # Max Drawdown

    # 2) running max, drawdown, 초과분 z_k
    for t in range(T):
        w_t = quicksum(x[i] * y.iloc[t, i] for i in range(N))               # 누적수익 w_t

        # Running Max 계산
        m.addConstr(u[t] >= w_t,          name=f"u_ge_w_{t}")
        if t > 0:
            m.addConstr(u[t] >= u[t-1],   name=f"u_monotone_{t}")

        # Drawdown 계산
        drawdown = u[t] - w_t
        m.addConstr(z[t] >= drawdown - z0, name=f"z_excess_{t}")            # z_k ≥ DD - z0

        # MaxDD 계산 제약
        m.addConstr(max_dd >= drawdown, name=f"maxdd_ge_dd_{t}")
    m.addConstr(max_dd <= gamma, name="maxdd_bound")
    # u_0 = 0
    m.addConstr(u[0] == 0, name="u0_zero")

    # 3) CDaR 제약 :  z0 + 1/((1-α)T) Σ z_k ≤ γ
    cdar_factor = 1 / ((1 - alpha) * T)
    m.addConstr(z0 + cdar_factor * quicksum(z[t] for t in range(T)) <= gamma,
                name="cdar_bound")

    # 4) 예산 (레버리지 금지)
    if leverage is None:
        m.addConstr(quicksum(x[i] for i in range(N)) == 1, name="budget")

    # 5) 목적함수 – 연환산 수익률 최대화
    mu = (1 + y.iloc[-1]) ** (252 / T) - 1
    m.setObjective(quicksum(x[i] * mu[i] for i in range(N)), GRB.MAXIMIZE)

    m.update()
    return m, x, u, z0, z, max_dd
