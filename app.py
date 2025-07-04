import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from io import BytesIO

@st.cache_data
def load_real_data():
    with open("tickers.txt", "r") as f:
        tickers = list(set([line.strip() for line in f if line.strip()]))

    end = datetime.date.today()
    start = end - datetime.timedelta(days=180)

    data = {}
    failed_download = []
    missing_close = []
    empty_frame = []
    success = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
            if df.empty:
                empty_frame.append(ticker)
                continue
            if 'Close' not in df.columns:
                missing_close.append(ticker)
                continue

            df.dropna(subset=['Close'], inplace=True)

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
            success.append(ticker)
        except Exception as e:
            print(f"❌ {ticker} 다운로드 실패: {e}")
            failed_download.append(ticker)
            continue

    return data, success, failed_download, empty_frame, missing_close

def evaluate_conditions(df):
    if df is None or df.empty or 'Close' not in df.columns:
        return {}, 0, "❌ 데이터 없음"

    latest = df.iloc[-1]
    recent_closes = df['Close'].tail(20)

    def safe_bool(val):
        if isinstance(val, (bool, np.bool_)):
            return val
        if pd.isnull(val):
            return False
        try:
            return bool(val)
        except:
            return False

    try:
        close_increase_5d = safe_bool(all(np.diff(recent_closes.tail(5)) > 0))
    except:
        close_increase_5d = False

    try:
        close_increase_4w = safe_bool(recent_closes.pct_change(20).iloc[-1] > 0)
    except:
        close_increase_4w = False

    conditions = {
        'avg_vol_above_500k': safe_bool(latest.get('VolumeAvg', 0) >= 500_000),
        'above_52w_low_30pct': safe_bool(latest['Close'] / df['Close'].min() >= 1.3),
        'close_up_5d': close_increase_5d,
        'close_up_4w': close_increase_4w,
        'above_ma20': safe_bool(latest['Close'] > latest.get('MA20', 0)),
        'ma20>ma50>ma200': safe_bool(latest.get('MA20', 0) > latest.get('MA50', 0) > latest.get('MA200', 0)),
        'rsi_ok': safe_bool(latest.get('RSI', 70) < 70),
        'macd_cross': safe_bool(latest.get('MACD', 0) > latest.get('MACD_signal', 0)),
        'stoch_cross': safe_bool(latest.get('Stoch_K', 0) > latest.get('Stoch_D', 0)),
        'growth_5y': True,
        'inst_buy': True
    }

    bool_values = [safe_bool(v) for v in conditions.values()]
    all_pass = all(bool_values)
    score = sum(bool_values)
    signal = "✅ 강한 매수 고려" if all_pass else "❌ 제외 (필수 조건 미충족)"
    return conditions, score, signal

# ------------------------ STREAMLIT UI ------------------------
real_data, success, failed, empty, no_close = load_real_data()
st.title("📈 전략형 주식 스크리너 (tickers.txt 기반)")

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

    buffer = BytesIO()
    summary_df.to_excel(buffer, index=False, engine='openpyxl')
    st.download_button(
        "📥 추천 종목 다운로드 (Excel)",
        buffer.getvalue(),
        "recommendations.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("오늘은 조건을 충족하는 종목이 없습니다.")

# ------------------------ 결과 정리 ------------------------
st.subheader("📁 티커 다운로드 상태 요약")
st.markdown(f"✅ 성공: {len(success)}개")
st.markdown(f"❌ 실패: {len(failed)}개")
st.markdown(f"⚠️ 빈 데이터: {len(empty)}개")
st.markdown(f"⚠️ 'Close' 없음: {len(no_close)}개")

with st.expander("❌ 실패한 티커 목록"):
    st.write(failed)

with st.expander("📉 데이터가 비어있는 티커"):
    st.write(empty)

with st.expander("🔍 'Close' 컬럼이 누락된 티커"):
    st.write(no_close)
