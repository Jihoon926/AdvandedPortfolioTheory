import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# tickers = ['KO', 'JNJ', 'XOM', 'MSFT', 'DUK', 'KHC', 'TLT', 'SHY', 'LQD', 'ILS']
# tickers = ['TSLA', 'AAPL', 'GOOG', 'MSFT', 'META', 'NVDA', 'TLT', 'SHY', 'BND', 'ILS']
tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "ORCL", "CRM", "AVGO", "ADBE"]
tickers = {"big_tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "ORCL", "CRM", "AVGO", "ADBE"], 
           "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO", "OXY", "HES"], 
           "finance":["JPM", "BAC", "WFC", "C", "MS", "GS", "USB", "PNC", "AXP", "SCHW"], 
           "HealthCare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "MDT", "BMY", "AMGN"], 
           "consumer_staples": ["KO", "PEP", "PG", "WMT", "COST", "MDLZ", "CL", "MO", "PM", "KR"],
           "industries": ["CAT", "HON", "RTX", "GE", "UNP", "BA", "ETN", "DE", "LMT", "ADP"], 
           "utilities": ["NEE", "SO", "DUK", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PEG"],
           "real_estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "WELL", "VTR", "O", "DLR"],
           "materials": ["LIN", "FCX", "APD", "ECL", "NEM", "DOW", "DD", "ALB", "BALL", "MLM"]
}

for sector, tickers in tickers.items():
    df = pd.DataFrame(columns=tickers)
    for ticker in tickers:
        stock_data = yf.Ticker(ticker)
        hist = stock_data.history(period="1y", auto_adjust=False)["Adj Close"]
        df[ticker] = hist
    correlation_matrix = df.corr()
    print(f"Correlation matrix for {sector}:")
    print(correlation_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.savefig(f"correlation_matrix_{sector}.png")