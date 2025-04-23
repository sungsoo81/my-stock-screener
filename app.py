import streamlit as st
import pandas as pd
import numpy as np
import datetime
import smtplib
from email.mime.text import MIMEText
from io import StringIO
import os

# ------------------------ MOCK DATA ------------------------
@st.cache_data

def load_mock_data():
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    dates = pd.date_range(end=datetime.date.today(), periods=126, freq='B')

    mock_data = {}
    for ticker in tickers:
        price = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        volume = np.random.randint(300_000, 2_000_000, len(dates))
        df = pd.DataFrame({
            'Date': dates,
            'Close': price,
            'High': price + np.random.uniform(0, 2, len(dates)),
            'Low': price - np.random.uniform(0, 2, len(dates)),
            'Open': price + np.random.uniform(-1, 1, len(dates)),
            'Volume': volume,
        }).set_index('Date')

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
        mock_data[ticker] = df
    return mock_data

# ------------------------ FILTERING ------------------------
def evaluate_conditions(df):
    latest = df.iloc[-1]
    recent_closes = df['Close'].tail(20)
    close_increase_5d = all(np.diff(recent_closes.tail(5)) > 0)
    close_increase_4w = recent_closes.pct_change(20).iloc[-1] > 0

    conditions = {
        'avg_vol_above_500k': latest['VolumeAvg'] >= 500_000,
        'above_52w_low_30pct': (latest['Close'] / df['Close'].min()) >= 1.3,
        'close_up_5d': close_increase_5d,
        'close_up_4w': close_increase_4w,
        'above_ma20': latest['Close'] > latest['MA20'],
        'ma20>ma50>ma200': (latest['MA20'] > latest['MA50']) and (latest['MA50'] > latest['MA200'] if not pd.isna(latest['MA200']) else False),
        'rsi_ok': latest['RSI'] < 70,
        'macd_cross': latest['MACD'] > latest['MACD_signal'],
        'stoch_cross': latest['Stoch_K'] > latest['Stoch_D'],
        'growth_5y': True,
        'inst_buy': True
    }

    all_pass = all(conditions.values())
    score = sum(conditions.values())
    signal = "✅ 강한 매수 고려" if all_pass else "❌ 제외 (필수 조건 미충족)"
    return conditions, score, signal

# ------------------------ STREAMLIT UI ------------------------
mock_data = load_mock_data()
st.title("📈 전략형 미국 주식 스크리너 (S&P 500, NASDAQ, Russell 2000)")

summary = []
today = datetime.date.today().isoformat()
log_file = "daily_recommendations.csv"

for ticker, df in mock_data.items():
    conds, score, signal = evaluate_conditions(df)
    if signal == "✅ 강한 매수 고려":
        colored_conds = {k: '✅' if v else '❌' for k, v in conds.items()}
        latest_close = df['Close'].iloc[-1]
        today_open = df['Open'].iloc[-1]
        change_from_open = (latest_close - today_open) / today_open * 100
        future_5d = df['Close'].shift(-5).iloc[-1] if len(df) >= 131 else np.nan
        future_10d = df['Close'].shift(-10).iloc[-1] if len(df) >= 136 else np.nan
        profit_5d = ((future_5d - latest_close) / latest_close * 100) if not np.isnan(future_5d) else np.nan
        profit_10d = ((future_10d - latest_close) / latest_close * 100) if not np.isnan(future_10d) else np.nan

        row = {
            'Date': today,
            'Ticker': ticker,
            'Score': score,
            'Signal': signal,
            'Price': round(latest_close, 2),
            'Change From Open (%)': round(change_from_open, 2),
            'Profit_5d (%)': round(profit_5d, 2) if profit_5d is not None else None,
            'Profit_10d (%)': round(profit_10d, 2) if profit_10d is not None else None,
            **colored_conds
        }
        summary.append(row)

summary_df = pd.DataFrame(summary)
if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
    log_df = pd.concat([log_df, summary_df], ignore_index=True)
else:
    log_df = summary_df.copy()
log_df.to_csv(log_file, index=False)

st.subheader("📊 오늘의 추천 종목")

if not summary_df.empty and 'Change From Open (%)' in summary_df.columns:
    st.dataframe(
        summary_df.style.applymap(
            lambda v: 'color: green' if isinstance(v, (int, float)) and v > 0
            else 'color: red' if isinstance(v, (int, float)) and v < 0 else None,
            subset=['Change From Open (%)']
        ),
        use_container_width=True
    )
else:
    st.dataframe(summary_df, use_container_width=True)

csv = summary_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
st.download_button("📥 결과 다운로드 (CSV)", csv, "screener_results.csv", "text/csv")

# 이메일 알림
if not summary_df.empty:
    from email.mime.text import MIMEText
    def send_email(subject, body, to="sungsoo81@gmail.com"):
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = "notifier@example.com"
        msg["To"] = to
        print(f"이메일 전송됨 → {to}\n제목: {subject}\n내용: {body}")

    send_email("📈 매수 추천 종목 있음", "오늘 매수 고려 종목이 발견되었습니다.")

elif not summary_df.empty and 'Change From Open (%)' in summary_df.columns and any(summary_df['Change From Open (%)'] < -5):
    send_email("📉 매도 경고 발생", "일부 종목이 당일 기준 -5% 이상 하락하였습니다. 주의하세요.")

# 상세 분석
if not summary_df.empty:
    selected = st.selectbox("📌 종목 상세 보기", summary_df['Ticker'])
    if selected:
        st.subheader(f"{selected} - 차트 및 시그널 분석")
        df = mock_data[selected].copy()
        st.line_chart(df[['Close', 'MA20', 'MA50', 'MA200']].dropna())
        with st.expander("📉 RSI / MACD / Stochastic 변화 추이 보기"):
            st.line_chart(df[['RSI']].dropna())
            st.line_chart(df[['MACD', 'MACD_signal']].dropna())
            st.line_chart(df[['Stoch_K', 'Stoch_D']].dropna())

        conds, score, signal = evaluate_conditions(df)
        st.markdown(f"### 시그널: **{signal}**")
        with st.expander("조건 세부 내용"):
            st.json({k: ('✅' if v else '❌') for k, v in conds.items()})
