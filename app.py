%%writefile app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

@st.cache
def download_data(symbol):
    data = yf.download(symbol, start='2022-01-01', end=pd.to_datetime('today').strftime('%Y-%m-%d'))
    data['Change'] = data['Close'].diff()
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    data.dropna(inplace=True)
    data['Label'] = data['Change'].apply(lambda x: 'Naik' if x > 0 else ('Turun' if x < 0 else 'Tetap'))
    return data

def train_model(data):
    features = data[['Close', 'MA_5', 'MA_10', 'RSI']]
    labels = data['Label']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, scaler, accuracy, report

def predict_by_date_range(start_date, end_date, data, model, scaler, coin):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date not in data.index or end_date not in data.index:
        st.error("Data tidak tersedia untuk salah satu atau kedua tanggal ini.")
        return

    filtered_data = data.loc[start_date:end_date]

    if filtered_data.empty:
        st.error("Data tidak tersedia dalam rentang tanggal yang diberikan.")
        return

    predictions = []

    for date in filtered_data.index:
        features_input = filtered_data.loc[date, ['Close', 'MA_5', 'MA_10', 'RSI']].values.reshape(1, -1)
        features_input_scaled = scaler.transform(features_input)

        prediction = model.predict(features_input_scaled)
        prediction_proba = model.predict_proba(features_input_scaled)

        predictions.append((date, prediction[0], prediction_proba[0]))

    dates, labels, probs = zip(*predictions)

    plt.figure(figsize=(14, 7))
    plt.plot(filtered_data['Close'], label='Close Price')

    for i, date in enumerate(dates):
        if labels[i] == 'Naik':
            plt.scatter(date, filtered_data.loc[date, 'Close'], color='green', marker='^', label='Naik' if i == 0 else "")
        else:
            plt.scatter(date, filtered_data.loc[date, 'Close'], color='red', marker='v', label='Turun' if i == 0 else "")

    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(f'{coin} Price with Predictions ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})')
    plt.legend()
    plt.grid(True)
    st.pyplot()

    for pred in predictions:
        st.write(f"Prediksi untuk tanggal {pred[0].strftime('%Y-%m-%d')}: {pred[1]}")
        st.write(f"Probabilitas: Naik = {pred[2][1]:.2f}, Turun = {pred[2][0]:.2f}")

def main():
    st.title('Crypto Price Prediction')

    coins = [
        'BTC-USD', 'ETH-USD', 'ADA-USD', 'XRP-USD', 'LTC-USD',
        'BCH-USD', 'LINK-USD', 'DOT-USD', 'BNB-USD', 'SOL-USD',
        'DOGE-USD', 'SHIB-USD', 'PEPE-USD', 'MATIC-USD', 'UNI-USD'
    ]

    coin = st.selectbox('Select Cryptocurrency', coins)

    start_date = st.date_input('Start Date', pd.to_datetime('2022-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('today'))

    data = download_data(coin)

    if st.button('Train Model'):
        model, scaler, accuracy, report = train_model(data)
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write('Classification Report:')
        st.write(report)

    if st.button('Predict'):
        if 'model' not in st.session_state or 'scaler' not in st.session_state:
            st.error("Model belum dilatih. Harap latih model terlebih dahulu.")
        else:
            predict_by_date_range(start_date, end_date, data, st.session_state['model'], st.session_state['scaler'], coin)

if __name__ == '__main__':
    main()

