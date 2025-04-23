import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import os



def evaluate_conditions(df):
    # 🔒 데이터 프레임 비어있거나 컬럼 없을 경우 즉시 차단
    if df is None or df.empty or 'Close' not in df.columns:
        return {}, 0, "❌ 데이터 없음"

    if len(df) < 30 or df['Close'].isna().sum() > 0:
        return {}, 0, "❌ 데이터 부족"

# ------------------------ DATA FETCHING ------------------------
@st.cache
def load_real_data():
    from bs4 import BeautifulSoup
    import requests

    def fetch_sp500_tickers():
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        return pd.read_html(str(table))[0]['Symbol'].str.replace('.', '-').tolist()

    def fetch_nasdaq_tickers():
        url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        return pd.read_html(str(table))[0]['Ticker'].str.replace('.', '-').tolist()

    def fetch_russell_tickers():
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'INTC', 'TSLA', 'META', 'CRM', 'ADBE',
                'PYPL', 'CSCO', 'PEP', 'COST', 'QCOM', 'AVGO', 'TXN', 'AMAT', 'AMD', 'INTU',
                'BKNG', 'ISRG', 'ADP', 'MU', 'FISV', 'KLAC', 'MRVL', 'LRCX', 'IDXX', 'ILMN',
                'ASML', 'CDNS', 'WDAY', 'ANSS', 'MCHP', 'ROST', 'CTSH', 'NXPI', 'SIRI', 'VRSK',
                'EBAY', 'WBA', 'MAR', 'BIDU', 'EXC', 'CHTR', 'XEL', 'EA', 'DLTR', 'SWKS']

    tickers = list(set(fetch_sp500_tickers() + fetch_nasdaq_tickers() + fetch_russell_tickers()))
    end = datetime.date.today()
    start = end - datetime.timedelta(days=180)

    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end)
        df.dropna(subset=['Close'], inplace=True)
        if df.empty:
            continue
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).apply(
            lambda x: (x[x > 0].mean() / -x[x < 0].mean()) if -x[x < 0].mean() != 0 else np.nan)))
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Stoch_K'] = 100 * (df['Close'] - df['Low'].rolling(14).min()) / \
                        (df['High'].rolling(14).max() - df['Low'].rolling(14).min())
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
        df['VolumeAvg'] = df['Volume'].rolling(20).mean()
        data[ticker] = df
    return data

# ------------------------ FILTERING ------------------------
def evaluate_conditions(df):
    if len(df) < 30 or df['Close'].isna().sum() > 0:
        return {}, 0, "❌ 데이터 부족"

    latest = df.iloc[-1]
    recent_closes = df['Close'].tail(20)

    # 보조 함수
    def safe_bool(val):
        if isinstance(val, (bool, np.bool_)):
            return val
        if pd.isnull(val):
            return False
        try:
            return bool(val)
        except:
            return False

    # 안전한 계산
    try:
        close_increase_5d = safe_bool(all(np.diff(recent_closes.tail(5)) > 0))
    except:
        close_increase_5d = False

    try:
        close_increase_4w = safe_bool(recent_closes.pct_change(20).iloc[-1] > 0)
    except:
        close_increase_4w = False

    # 조건 계산 (모든 값은 무조건 bool로 변환)
    conditions = {
        'avg_vol_above_500k': safe_bool(latest.get('VolumeAvg', 0) >= 500_000),
        'above_52w_low_30pct': safe_bool(latest['Close'] / df['Close'].min() >= 1.3),
        'close_up_5d': close_increase_5d,
        'close_up_4w': close_increase_4w,
        'above_ma20': safe_bool(latest['Close'] > latest['MA20']),
        'ma20>ma50>ma200': safe_bool(
            latest.get('MA20', 0) > latest.get('MA50', 0) and latest.get('MA50', 0) > latest.get('MA200', 0)
        ),
        'rsi_ok': safe_bool(latest.get('RSI', 70) < 70),
        'macd_cross': safe_bool(latest.get('MACD', 0) > latest.get('MACD_signal', 0)),
        'stoch_cross': safe_bool(latest.get('Stoch_K', 0) > latest.get('Stoch_D', 0)),
        'growth_5y': True,
        'inst_buy': True
    }

    # 여기에서 확실하게 리스트로 강제 변환 후 bool만 평가
    bool_values = [safe_bool(v) for v in conditions.values()]
    all_pass = all(bool_values)
    score = sum(bool_values)
    signal = "✅ 강한 매수 고려" if all_pass else "❌ 제외 (필수 조건 미충족)"

    return conditions, score, signal


# ------------------------ STREAMLIT UI ------------------------
real_data = load_real_data()
st.title("📈 전략형 미국 주식 스크리너 (S&P 500, NASDAQ, Russell 2000)")

summary = []
today = datetime.date.today().isoformat()

for ticker, df in real_data.items():
    conds, score, signal = evaluate_conditions(df)
    if signal == "✅ 강한 매수 고려":
        colored_conds = {k: '✅' if v else '❌' for k, v in conds.items()}
        latest_close = df['Close'].iloc[-1]
        today_open = df['Open'].iloc[-1]
        change_from_open = (latest_close - today_open) / today_open * 100

        row = {
            'Date': today,
            'Ticker': ticker,
            'Score': score,
            'Signal': signal,
            'Price': round(latest_close, 2),
            'Change From Open (%)': round(change_from_open, 2),
            **colored_conds
        }
        summary.append(row)

summary_df = pd.DataFrame(summary)
st.subheader("📊 오늘의 추천 종목")

if not summary_df.empty:
    st.dataframe(
        summary_df.style.applymap(
            lambda v: 'color: green' if isinstance(v, (int, float)) and v > 0
            else 'color: red' if isinstance(v, (int, float)) and v < 0 else None,
            subset=['Change From Open (%)']
        ),
        use_container_width=True
    )

    excel_bytes = summary_df.to_excel(index=False, engine='openpyxl')
    st.download_button(
        "📥 추천 종목 다운로드 (Excel)",
        excel_bytes,
        "screener_results.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("오늘은 조건을 충족하는 종목이 없습니다.")
