import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time
from statsmodels.api import OLS, add_constant
import seaborn as sns
from adjustText import adjust_text

def SP_data():
    # S&P500 종목 리스트 크롤링
    print("S&P500 종목 리스트 가져오는 중...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)
    sp500_df = sp500_table[0]
    sp500_df = sp500_df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]
    sp500_df.columns = ['Ticker', 'Name', 'Sector', 'Subsector']

    # 시가총액 가져오는 함수 정의
    def get_market_cap(ticker):
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap', None)
            return market_cap
        except:
            return None

    # 전체 종목에 대해 시가총액 가져오기
    print("시가총액 데이터 가져오는 중...(시간 약간 걸려요)")
    market_caps = []
    for ticker in list(sp500_df['Ticker']) + ["EWJ", "EWY", "FXI", "FEZ", "GLD", "GSG", "SLV", "TLT", "SHY", "BIL"]:
        market_cap = get_market_cap(ticker)
        market_caps.append(market_cap)
        #time.sleep(0.5)  # 과부하 방지 위해 0.5초 쉬기


    tickers = ["SPY", "EWJ", "EWY", "FXI", "FEZ", "GLD", "GSG", "SLV", "TLT", "SHY", "BIL"]
    names = ["S&P500 Index", "iShares MSCI Japan ETF", "MSCI South Korea Cap ETF", "iShares China Large-Cap ETF", "SPDR Euro Stoxx 50 ETF", "SPDR Gold Trust"
            , "iShares S&P GSCI Commodity-Indexed Trust ETF", "iShares Silver Trust", "iShares 20+ Year Treasury Bond ETF"
            , "Vanguard Total Bond Market Index Fund ETF", "iShares 0-3 Month Treasury Bond ETF"]
    sectors = ["Index", "Foreign Market Index", "Foreign Market Index", "Foreign Market Index", "Foreign Market Index"
            , "Commodity Index", "Commodity Index", "Commodity Index"
            , "Bond", "Bond", "Bond"]
    subsectors = ["Index", "Japan", "Korea", "China", "Euro-zone", "Gold", "Composite Commodity", "Silver"
                , "Long-term Bond", "Composite Bond", "Short-term Bond"]

    for ticker, name, sector, subsector in zip(tickers, names, sectors, subsectors):
        sp500_df.loc[len(sp500_df)] = ([ticker, name, sector, subsector])
        
    if len(market_caps) < len(sp500_df):
        missing_count = len(sp500_df) - len(market_caps)
        market_caps.extend([None] * missing_count)


    sp500_df['MarketCap'] = market_caps
    sp500_df.to_csv('sp500.csv', index=False)

    return sp500_df

def return_data(sp500_df):
    tickers = sp500_df['Ticker'].tolist()

    batch_size = 50
    prices_5yr = []
    count = 0

    print("📥 데이터 다운로드 시작...")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"\n📦 Batch {i // batch_size + 1}: Downloading {len(batch)} tickers")

        try:
            data = yf.download(batch, start="2020-04-25", end="2025-04-24",
                            auto_adjust=False, progress=False)['Adj Close']
            
            if isinstance(data, pd.Series):  # 단일 종목만 리턴된 경우 처리
                data = data.to_frame()

            for ticker in batch:
                if ticker in data.columns:
                    price = data[[ticker]]
                    prices_5yr.append(price)
                    count += 1
                    print(f"✅ {ticker} 완료 ({count}/{len(tickers)})")
                else:
                    print(f"⚠️ {ticker} → 데이터 없음")

        except Exception as e:
            print(f"⛔️ Error in batch {i // batch_size + 1}: {e}")

        print("⏳ 10초 대기 중...")
        time.sleep(10)

    return_df = pd.concat(prices_5yr, axis=1)
    return_df.to_csv('out/return_df.csv')

    return return_df

def selected_data(returns_df, info_df):
    # returns_df = pd.read_csv("return_df.csv", parse_dates=["Date"]).rename(columns={"Date": "date"})
    # info_df = pd.read_csv("sp500.csv")

    # 티커 목록 정제
    tickers = [col for col in returns_df.columns if col != "date"]
    snp500_tickers = list(set(tickers) & set(info_df['Ticker']))
    snp500_tickers = sorted(snp500_tickers)[:-10]  # 마지막 10개 제외

    # 섹터별 시가총액 비례 계산
    info_filtered = info_df[info_df['Ticker'].isin(snp500_tickers)].copy()
    sector_weights = info_filtered.groupby("Sector")["MarketCap"].sum()
    sector_counts = ((sector_weights / sector_weights.sum()) * 40).round().astype(int)

    # 합계가 40이 안 맞을 경우 조정
    while sector_counts.sum() != 40:
        diff = 40 - sector_counts.sum()
        target = sector_counts.idxmax() if diff < 0 else sector_counts.idxmin()
        sector_counts[target] += np.sign(diff)

    # 수익률 계산
    returns_df.set_index("date", inplace=True)
    returns_df.drop('Unnamed: 0', axis=1, inplace=True)
    returns_df = returns_df.dropna(axis=1, how='all')
    returns_df = returns_df.ffill(axis=1)
    returns = returns_df.pct_change().dropna()

    # 시장 수익률 (SPY 사용)
    market = returns["SPY"]

    # 무위험 수익률 예시 (수정 가능): 연 4% 고정 → 일간 수익률
    rf_daily = 0.04 / 252
    rf = pd.Series(rf_daily, index=returns.index)

    # 결과 저장
    result_list = []

    for ticker in snp500_tickers:

        if ticker in ['GOOG', 'NWSA']:
            continue
        
        try:
            stock = returns[ticker].dropna()
            aligned = pd.concat([stock, market, rf], axis=1, join="inner").dropna()
            if len(aligned) < 200:
                continue  # 데이터 너무 적으면 제외
            r_i = aligned[ticker]
            r_m = aligned["SPY"]
        
            excess_m = r_m - aligned[0]
            excess_m.name = "excess_market"
            excess_i = r_i - aligned[0]
            
            X = add_constant(excess_m)
            model = OLS(excess_i, X).fit()
            alpha = model.params["const"]
            beta = model.params["excess_market"]
        
            X = add_constant(excess_m)
            model = OLS(excess_i, X).fit()
            alpha = model.params["const"]
            beta = model.params[excess_m.name]
            r_squared = model.rsquared
            volatility = r_i.std() * np.sqrt(252)  # 연율화
        
            result_list.append({
                "Ticker": ticker,
                "Alpha": alpha,
                "Beta": beta,
                "R2": r_squared,
                "Volatility": volatility,
            })
        except KeyError:
            continue
        
    result_df = pd.DataFrame(result_list)
    merged = result_df.merge(info_df, on="Ticker")

    # 필터링: R² 기준
    filtered = merged[merged["R2"] >= 0.25].copy()

    # 섹터별 종목 선정
    final_selected = []

    for sector, count in sector_counts.items():
        group = filtered[filtered["Sector"] == sector].copy()
        if group.empty:
            continue
        # 스코어 계산: 알파 높고, 변동성 낮고, 베타 1에 가까운 순
        group["Score"] = (
            group["MarketCap"].rank(ascending=False) * 0.4 +
            group["Alpha"].rank(ascending=False) * 0.3 +
            group["Volatility"].rank(ascending=True) * 0.2 +
            (group["Beta"] - 1).abs().rank(ascending=True) * 0.1
        )
        final_selected.extend(group.sort_values("Score").head(count)["Ticker"].tolist())

    # 마지막 10개 ETF 또는 비대상 종목 추가
    excluded_10 = tickers[-10:]

    # 최종 선정 종목 리스트 (40 + 10 = 50개)
    final_50 = final_selected + excluded_10

    print("최종 선정된 50개 종목:")
    print(final_50)

    info_df[info_df['Ticker'].isin(final_50)].sort_values(['Sector', 'MarketCap'], ascending=True)

    selected_price_data = returns_df[['Date'] + final_50]
    selected_price_data.to_csv('selected_price_data.csv', index=False)

    selected_meta_data = info_df[info_df['Ticker'].isin(final_50)]
    selected_meta_data.to_csv('selected_meta_data.csv', index=False)


    return info_df, filtered, final_50

def expected_returns_plot(filtered):
        # 예시 파라미터 (수정 가능)
    rf = 0.04  # 무위험 수익률
    rm = 0.10  # 시장 기대 수익률

    # SML 선 계산
    min_beta = filtered["Beta"].min()
    max_beta = filtered["Beta"].max()
    beta_range = np.linspace(min_beta - 0.2, max_beta + 0.2, 100)
    sml_y = rf + beta_range * (rm - rf)

    # 종목의 기대 수익률 계산: alpha + rf
    filtered["ExpectedReturn"] = filtered["Alpha"]*252 + rf

    # 시각화
    plt.figure(figsize=(12, 7))
    # SML 선
    plt.plot(beta_range, sml_y, label="Security Market Line (SML)", color='black', linestyle='--')
    # 종목 점
    sns.scatterplot(data=filtered, x="Beta", y="ExpectedReturn", hue="Sector", size="MarketCap", palette="tab10", legend=False)

    # 시각 보조선
    plt.axvline(1.0, color='gray', linestyle=':', label='Beta = 1')
    plt.title("CAPM Alpha + Beta with Security Market Line")
    plt.xlabel("Beta")
    plt.ylabel("Expected Return (Alpha + Rf)")
    plt.grid(True)
    plt.legend()
    plt.show()

def alpha_beta(filtered, final_50):
    # 예시 파라미터 (수정 가능)
    rf = 0.04  # 무위험 수익률
    rm = 0.10  # 시장 기대 수익률

    # SML 선 계산
    min_beta = filtered["Beta"].min()
    max_beta = filtered["Beta"].max()
    beta_range = np.linspace(min_beta - 0.2, max_beta + 0.2, 100)
    sml_y = rf + beta_range * (rm - rf)

    # 종목의 기대 수익률 계산: alpha + rf
    filtered["ExpectedReturn"] = filtered["Alpha"]*252 + rf

    # final_40 리스트 (컬러로 나타낼 종목)
    final_40 = final_50[:-10]

    # 종목을 색상으로 구분 (final_40에 포함된 종목은 컬러로, 나머지는 흑백으로)
    filtered['color'] = filtered['Ticker'].apply(lambda x: 'red' if x in final_40 else 'gray')

    # 시각화
    plt.figure(figsize=(12, 7))

    # SML 선
    plt.plot(beta_range, sml_y, label="Security Market Line (SML)", color='black', linestyle='--')

    # 종목 점 (색상 적용)
    sns.scatterplot(data=filtered, x="Beta", y="ExpectedReturn", size="MarketCap", hue="color", palette={"red": "red", "gray": "gray"}, alpha=0.5, legend=False)

    # final_40에 포함된 종목만 각 점에 ticker를 표시
    texts = []  # 텍스트 객체를 저장할 리스트
    for i in range(len(filtered)):
        ticker = filtered.iloc[i]['Ticker']
        if ticker in final_40:  # final 40에 있는 경우에만 텍스트 표시
            beta = filtered.iloc[i]['Beta']
            expected_return = filtered.iloc[i]['ExpectedReturn']
            text  = plt.text(beta+0.01, expected_return+0.01, ticker, fontsize=7, ha='center', va='center')
            texts.append(text)  # 텍스트 객체를 리스트에 추가

    # 텍스트 겹침 방지
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray'))

    # 시각 보조선
    plt.axvline(1.0, color='gray', linestyle=':', label='Beta = 1')

    # 그래프 제목 및 라벨
    plt.title("CAPM Alpha + Beta with Security Market Line")
    plt.xlabel("Beta")
    plt.ylabel("Expected Return (Alpha + Rf)")
    plt.grid(True)

    # 범례
    # plt.legend(title='Ticker', labels=final_40)

    plt.show()