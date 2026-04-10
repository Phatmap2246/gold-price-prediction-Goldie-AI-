import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import gspread
from google.colab import drive, auth
from google.auth import default
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from fredapi import Fred  # nếu dùng fredapi

# Load API keys từ .env
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

# Đảm bảo đã tải lexicon cho VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ================================================
# Ô 4: PIPELINE CHÍNH - CÀO, HUẤN LUYỆN, DỰ ĐOÁN, LƯU KẾT QUẢ
# ================================================

# --- Khai báo các hằng số ---
DRIVE_FOLDER_PATH = "/content/drive/MyDrive/Goldie_AI/"
os.makedirs(DRIVE_FOLDER_PATH, exist_ok=True)
MODEL_FILE_PATH = os.path.join(DRIVE_FOLDER_PATH, "gold_lstm_multivariate.h5")
CHART_FILE_PATH = os.path.join(DRIVE_FOLDER_PATH, "latest_gold_forecast.png")

# --- 1. Lấy dữ liệu giá vàng ---
print("1. Lấy dữ liệu giá vàng...")
df_gold = yf.Ticker("GLD").history(period="max")
df_gold = df_gold[['Close']].rename(columns={'Close': 'gold'})

# --- 2. Lấy dữ liệu lãi suất FED (từ FRED) ---
print("2. Lấy dữ liệu lãi suất FED...")
fred = Fred(api_key=FRED_API_KEY)
fed_rates = fred.get_series('DFF')  # Lấy tất cả
fed_rates = fed_rates.resample('D').ffill()  # Nội suy sang hàng ngày
fed_df = pd.DataFrame(fed_rates, columns=['fed_rate'])

# --- 3. Lấy tin tức và tính sentiment hàng ngày ---
print("3. Lấy tin tức và tính sentiment...")
end_date = pd.Timestamp.now().date()
start_date = end_date - pd.Timedelta(days=30)  # lấy 30 ngày gần nhất

def fetch_daily_sentiment(date):
    date_str = date.strftime('%Y-%m-%d')
    query = f"Federal Reserve gold after:{date_str} before:{date_str}"
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWSAPI_KEY}&language=en&pageSize=10"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return 0
        articles = response.json().get('articles', [])
        if not articles:
            return 0
        scores = []
        for art in articles:
            title = art.get('title', '')
            if title:
                scores.append(sia.polarity_scores(title)['compound'])
        return np.mean(scores) if scores else 0
    except:
        return 0

date_range = pd.date_range(start=start_date, end=end_date)
sentiments = []
for d in date_range:
    sent = fetch_daily_sentiment(d)
    sentiments.append(sent)
    print(f"  {d.strftime('%Y-%m-%d')}: {sent:.2f}")

sentiment_df = pd.DataFrame({'date': date_range, 'sentiment': sentiments})
sentiment_df.set_index('date', inplace=True)

# --- 4. Ghép dữ liệu ---
print("4. Ghép dữ liệu...")
df = df_gold.join(fed_df, how='inner').join(sentiment_df, how='inner')
df = df.dropna()
print(f"Kết quả: {len(df)} ngày dữ liệu")
print(df.tail())

# --- 5. Chuẩn bị dữ liệu cho LSTM ---
print("5. Chuẩn bị dữ liệu LSTM...")
features = ['gold', 'fed_rate', 'sentiment']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

seq_len = 60
X, y = [], []
for i in range(seq_len, len(scaled_data)):
    X.append(scaled_data[i-seq_len:i])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- 6. Huấn luyện mô hình ---
print("6. Huấn luyện mô hình...")
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(seq_len, len(features))),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), verbose=1)

model.save(MODEL_FILE_PATH)
print(f"Đã lưu mô hình vào {MODEL_FILE_PATH}")

# --- 7. Dự đoán và đánh giá ---
print("7. Dự đoán và vẽ biểu đồ...")
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(
    np.concatenate([predictions, np.zeros((len(predictions), len(features)-1))], axis=1)
)[:,0]

y_test_real = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), len(features)-1))], axis=1)
)[:,0]

mae = mean_absolute_error(y_test_real, predictions)
rmse = np.sqrt(mean_squared_error(y_test_real, predictions))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

plt.figure(figsize=(16,8))
plt.plot(y_test_real, label='Thực tế')
plt.plot(predictions, label='Dự đoán')
plt.title('Dự báo giá vàng - LSTM đa biến')
plt.xlabel('Mẫu test')
plt.ylabel('Giá vàng (USD)')
plt.legend()
plt.savefig(CHART_FILE_PATH)
print(f"Đã lưu biểu đồ vào {CHART_FILE_PATH}")

# --- 8. Dự đoán ngày mai ---
print("8. Dự đoán giá ngày mai...")
last_60 = scaled_data[-seq_len:]  # lấy 60 ngày cuối
X_future = np.array([last_60])
pred_scaled = model.predict(X_future)[0][0]
pred_array = np.zeros((1, len(features)))
pred_array[0,0] = pred_scaled
pred_price = scaler.inverse_transform(pred_array)[0,0]
real_last_price = df['gold'].iloc[-1]
print(f"Giá hôm nay: {real_last_price:.2f} USD")
print(f"Dự đoán ngày mai: {pred_price:.2f} USD")

# --- 9. Lưu kết quả vào Google Sheet (tùy chọn) ---
drive.mount('/content/drive')
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

try:
    sheet = gc.open("Goldie_Output").sheet1
except:
    print("Tạo sheet mới...")
    sh = gc.create("Goldie_Output")
    sheet = sh.sheet1
    sh.share('', perm_type='anyone', role='writer')

sheet.update_cell(1, 1, "Ký hiệu")
sheet.update_cell(1, 2, "Giá trị")
sheet.update_cell(2, 1, "Giá Hôm Nay")
sheet.update_cell(2, 2, f"{real_last_price:.2f}")
sheet.update_cell(3, 1, "Dự Đoán AI (Ngày mai)")
sheet.update_cell(3, 2, f"{pred_price:.2f}")
sheet.update_cell(4, 1, "Cập nhật lần cuối")
sheet.update_cell(4, 2, pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
sheet.update_cell(5, 1, "MAE")
sheet.update_cell(5, 2, f"{mae:.2f}")
sheet.update_cell(6, 1, "RMSE")
sheet.update_cell(6, 2, f"{rmse:.2f}")
sheet.update_cell(7, 1, "Tên file Biểu đồ")
sheet.update_cell(7, 2, "latest_gold_forecast.png")

print("✅ Hoàn tất pipeline! Kết quả đã được lưu vào Drive và Google Sheet.")
