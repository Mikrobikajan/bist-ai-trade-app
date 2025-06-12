import streamlit as st
import pandas as pd
import datetime
import investpy
from ta import momentum, trend, volatility, volume

# Verileri çekme fonksiyonları
def get_bist_stocks():
    stocks = investpy.stocks.get_stocks(country='turkey')
    return stocks['symbol'].tolist()

def get_stock_data(stock_symbol):
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=365)
    
    df = investpy.get_stock_historical_data(
        stock=stock_symbol,
        country='Turkey',
        from_date=start_date.strftime('%d/%m/%Y'),
        to_date=end_date.strftime('%d/%m/%Y')
    )

    df = df.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
    return df

def calculate_indicators(df):
    df['rsi'] = momentum.RSIIndicator(close=df['close']).rsi()
    macd = trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['ema'] = trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['sma'] = trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    boll = volatility.BollingerBands(close=df['close'])
    df['boll_upper'] = boll.bollinger_hband()
    df['boll_lower'] = boll.bollinger_lband()
    df['stochastic'] = momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()
    df['roc'] = momentum.ROCIndicator(close=df['close']).roc()
    df['obv'] = volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['cmf'] = volume.ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).chaikin_money_flow()
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    return df

def generate_signal(row):
    weights = {
        "rsi": 1.5,
        "macd": 2.0,
        "ema": 1.0,
        "sma": 1.0,
        "boll": 1.5,
        "stochastic": 1.0,
        "roc": 1.0,
        "obv": 1.0,
        "cmf": 1.0,
        "vwap": 1.0
    }

    score_al = 0
    score_sat = 0

    if row['rsi'] < 30:
        score_al += weights['rsi']
    elif row['rsi'] > 70:
        score_sat += weights['rsi']

    if row['macd'] > row['macd_signal']:
        score_al += weights['macd']
    else:
        score_sat += weights['macd']

    if row['close'] > row['ema']:
        score_al += weights['ema']
    else:
        score_sat += weights['ema']

    if row['close'] > row['sma']:
        score_al += weights['sma']
    else:
        score_sat += weights['sma']

    if row['close'] < row['boll_lower']:
        score_al += weights['boll']
    elif row['close'] > row['boll_upper']:
        score_sat += weights['boll']

    if row['stochastic'] < 20:
        score_al += weights['stochastic']
    elif row['stochastic'] > 80:
        score_sat += weights['stochastic']

    if row['roc'] > 0:
        score_al += weights['roc']
    else:
        score_sat += weights['roc']

    if row['obv'] > row['obv'].mean():
        score_al += weights['obv']
    else:
        score_sat += weights['obv']

    if row['cmf'] > 0:
        score_al += weights['cmf']
    else:
        score_sat += weights['cmf']

    if row['close'] > row['vwap']:
        score_al += weights['vwap']
    else:
        score_sat += weights['vwap']

    final = "BEKLE"
    if score_al >= 9:
        final = "AL"
    elif score_sat >= 9:
        final = "SAT"

    return final

def backtest_strategy(df, initial_cash):
    df['signal'] = df.apply(generate_signal, axis=1)
    df['next_close'] = df['close'].shift(-1)
    df['return'] = 0.0

    cash = initial_cash
    position = 0
    equity_curve = []

    for i in range(len(df) - 1):
        price_today = df.at[i, 'close']
        price_next = df.at[i + 1, 'close']
        signal = df.at[i, 'signal']

        if signal == 'AL' and cash > 0:
            position = cash / price_today
            cash = 0
        elif signal == 'SAT' and position > 0:
            cash = position * price_today
            position = 0

        total_value = cash + position * price_today
        equity_curve.append(total_value)

        if signal == 'AL':
            df.at[i, 'return'] = (price_next - price_today) / price_today
        elif signal == 'SAT':
            df.at[i, 'return'] = (price_today - price_next) / price_today
        else:
            df.at[i, 'return'] = 0.0

    df['equity'] = [initial_cash] + equity_curve
    total_return = (df['equity'].iloc[-2] - initial_cash) / initial_cash
    win_rate = (df['return'] > 0).sum() / (df['signal'].isin(['AL', 'SAT']).sum()) * 100
    avg_return = df['return'].mean() * 100

    return df, total_return, win_rate, avg_return

# Streamlit arayüzü
st.title("📈 BIST AI Trade Asistan")

stocks = get_bist_stocks()
selected_stock = st.selectbox("Bir hisse senedi seçin:", stocks)
initial_cash = st.number_input("Başlangıç sermayenizi girin (₺):", min_value=1000, max_value=1000000, value=100000, step=1000)

if st.button("🔄 Verileri Güncelle"):
    df = get_stock_data(selected_stock)
    df = calculate_indicators(df)
    df = df.dropna()
    latest = df.iloc[-1]
    karar = generate_signal(latest)

    st.subheader(f"Son Karar: {karar}")
    st.line_chart(df['close'], use_container_width=True)
    st.dataframe(df.tail(5).round(2))

    # Backtest Sonuçları
    df_bt, total_return, win_rate, avg_return = backtest_strategy(df, initial_cash)
    st.subheader("📊 Strateji Backtest Sonuçları")
    st.metric("Toplam Getiri (%)", f"{total_return * 100:.2f}")
    st.metric("İsabet Oranı (%)", f"{win_rate:.2f}")
    st.metric("Ortalama Getiri (%)", f"{avg_return:.2f}")
    st.line_chart(df_bt['equity'], use_container_width=True)
