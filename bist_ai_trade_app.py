import streamlit as st
import pandas as pd
import datetime
import requests
import os
from ta import momentum, trend, volatility, volume
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import traceback

# BIST sembollerini elle tanımlayabiliriz (yfinance'de .IS uzantısı kullanılır)
def get_bist_symbols_from_finnhub():
    return {
        'ASELS': 'ASELS.IS',
        'THYAO': 'THYAO.IS',
        'KRDMD': 'KRDMD.IS',
        'SISE': 'SISE.IS',
        'BIMAS': 'BIMAS.IS'
    }

BIST_SYMBOLS = get_bist_symbols_from_finnhub()

# Hisse listesini döndür
def get_bist_stocks():
    return list(BIST_SYMBOLS.keys())

# Belirli bir hisse için veriyi çek
# Artık Finnhub API kullanılıyor
import yfinance as yf

def get_stock_data(stock_symbol):
    yf_symbol = BIST_SYMBOLS[stock_symbol]
    df = yf.download(yf_symbol, period="1y", interval="1d")

    if df is None or df.empty:
        raise ValueError(f"{yf_symbol} için yfinance verisi bulunamadı.")

    df.rename(columns=str.lower, inplace=True)

    if 'close' not in df.columns:
        raise ValueError(f"{yf_symbol} verisinde 'close' sütunu bulunamadı.
Sütunlar: {df.columns.tolist()}")}")}")}")

    df.dropna(subset=['close'], inplace=True)
    return df

# Göstergeleri hesapla (güvenli versiyon)
def calculate_indicators(df):
    df = df.copy()
    if 'close' not in df.columns:
        raise ValueError("'close' sütunu eksik olduğu için göstergeler hesaplanamıyor.")
    df = df.dropna(subset=['close'])

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

# ML tabanlı sinyal üretici

def generate_ml_signal(df):
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna(subset=['target'])

    features = [
        'rsi', 'macd', 'macd_signal', 'ema', 'sma', 'boll_upper', 'boll_lower',
        'stochastic', 'roc', 'obv', 'cmf', 'vwap'
    ]

    df = df.dropna(subset=features + ['target'])

    if df.empty or df.shape[0] < 20:
        raise ValueError("Yeterli veri yok. Lütfen farklı bir hisse veya zaman aralığı seçin.")

    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.info(f"Model doğruluk oranı: %{accuracy * 100:.2f}")

    latest_features = df[features].iloc[-1:].copy()
    latest_features = latest_features.fillna(method='ffill').fillna(method='bfill')
    if latest_features.isnull().values.sum() > 0:
        raise ValueError("Son veri noktasında eksik gösterge verisi var.")

    prediction = model.predict(latest_features)[0]

    return "AL" if prediction == 1 else "SAT"

# Streamlit arayüzü
st.title("📈 BIST AI Trade Asistan - ML Versiyon")

stocks = get_bist_stocks()

if not stocks:
    st.error("Finnhub'dan BIST sembol listesi alınamadı. Lütfen API anahtarını ve limiti kontrol edin.")
    st.stop()

selected_stock = st.selectbox("Bir hisse senedi seçin:", stocks)

if st.button("🔄 Verileri Güncelle (ML Karar)"):
    try:
        df = get_stock_data(selected_stock)
        df = calculate_indicators(df)
        df = df.dropna()

        karar = generate_ml_signal(df)

        st.subheader(f"📌 Makine Öğrenmesi ile Tahmin: {karar}")
        st.line_chart(df['close'], use_container_width=True)
        st.dataframe(df.tail(5).round(2))

    except Exception as e:
        st.error("Veri alınamadı veya analizde hata oluştu.")
        st.text(traceback.format_exc())
