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
from fredapi import Fred

# Load API keys từ .env
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

# Đảm bảo đã tải lexicon cho VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ================================================
# PIPELINE CHÍNH - CÀO, HUẤN LUYỆN, DỰ ĐOÁN, LƯU KẾT QUẢ
# ================================================

# --- Khai báo các hằng số ---
DRIVE_FOLDER_PATH = "/content/drive/MyDrive/Goldie_AI/"
os.makedirs(DRIVE_FOLDER_PATH, exist_ok=True)
MODEL_FILE_PATH = os.path.join(DRIVE_FOLDER_PATH, "gold_lstm_multivariate.h5")
CHART_FILE_PATH = os.path.join(DRIVE_FOLDER_PATH, "latest_gold_forecast.png")

# --- 1. Lấy dữ liệu giá vàng (10 năm gần nhất) ---
print("1. Lấy dữ liệu giá vàng...")
df_gold = yf.Ticker("GLD").history(period="10y")
df_gold = df_gold[['Close']].rename(columns={'Close': 'gold'})

# --- 2. Lấy dữ liệu lãi suất FED và Chỉ số sợ hãi VIX ---
print("2. Lấy dữ liệu vĩ mô (FED Rate & VIX Index)...")
fred = Fred(api_key=FRED_API_KEY)
fed_rates = fred.get_series('DFF')
fed_rates = fed_rates.resample('D').ffill()
fed_df = pd.DataFrame(fed_rates, columns=['fed_rate'])

# Thêm chỉ số VIX để đo lường mức độ bất ổn thị trường
vix_data = yf.Ticker("^VIX").history(period="10y")[['Close']].rename(columns={'Close': 'vix'})
vix_data.index = vix_data.index.tz_localize(None)

# --- 3. Cào tin tức địa chính trị và tính Sentiment ---
print("3. Đang đánh giá mức độ lạc quan/căng thẳng thị trường...")
end_date = pd.Timestamp.now().date()
start_date = end_date - pd.Timedelta(days=30)

def fetch_daily_sentiment(date):
    date_str = date.strftime('%Y-%m-%d')
    # Nâng cấp query: Quét cả tin tức FED và các biến động địa chính trị
    query = "(Federal Reserve OR gold OR war OR conflict OR geopolitical) AND (gold OR economy)"
    url = f"https://newsapi.org/v2/everything?q={query}&from={date_str}&to={date_str}&apiKey={NEWSAPI_KEY}&language=en&pageSize=15"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return 0
        articles = response.json().get('articles', [])
        if not articles:
            return 0
        scores = []
        for art in articles:
            text_to_analyze = (art.get('title', '') + " " + art.get('description', '')).strip()
            if text_to_analyze:
                scores.append(sia.polarity_scores(text_to_analyze)['compound'])
        return np.mean(scores) if scores else 0
    except:
        return 0

date_range = pd.date_range(start=start_date, end=end_date)
sentiments = []
for d in date_range:
    sent = fetch_daily_sentiment(d)
    sentiments.append(sent)
    print(f"  {d.strftime('%Y-%m-%d')} Sentiment Score: {sent:.2f}")

sentiment_df = pd.DataFrame({'date': date_range, 'sentiment': sentiments})
sentiment_df.set_index('date', inplace=True)

# --- 4. Ghép dữ liệu đa biến ---
print("4. Ghép dữ liệu đa biến...")
df_gold.index = df_gold.index.tz_localize(None)
df = df_gold.join(fed_df, how='left').join(vix_data, how='left').join(sentiment_df, how='left')

# Xử lý dữ liệu thiếu bằng nội suy (Forward Fill)
df['fed_rate'] = df['fed_rate'].ffill().fillna(0)
df['vix'] = df['vix'].ffill().fillna(df['vix'].mean()) # VIX dùng trung bình nếu thiếu
df['sentiment'] = df['sentiment'].fillna(0)
df = df.dropna()

print(f"Kết quả: {len(df)} ngày dữ liệu sẵn sàng (Features: {df.columns.tolist()})")

# --- 5. Chuẩn bị dữ liệu cho LSTM (Chống Data Leakage) ---
print("5. Chuẩn bị dữ liệu LSTM...")
features = ['gold', 'fed_rate', 'vix', 'sentiment']
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]

scaler = MinMaxScaler()
scaler.fit(train_df[features])
scaled_data = scaler.transform(df[features])

seq_len = 60
X, y = [], []
for i in range(seq_len, len(scaled_data)):
    X.append(scaled_data[i-seq_len:i])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X_train, X_test = X[:split_idx-seq_len], X[split_idx-seq_len:]
y_train, y_test = y[:split_idx-seq_len], y[split_idx-seq_len:]

# --- 6. Huấn luyện mô hình đa biến ---
print("6. Huấn luyện mô hình Goldie AI...")
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(seq_len, len(features))),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test), verbose=1)

model.save(MODEL_FILE_PATH)

# --- 7. Dự đoán và đánh giá ---
print("7. Dự đoán và vẽ biểu đồ...")
predictions = model.predict(X_test)
# Nghịch đảo scale để lấy giá USD thực
pred_full = np.concatenate([predictions, np.zeros((len(predictions), len(features)-1))], axis=1)
predictions = scaler.inverse_transform(pred_full)[:,0]

y_real_full = np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), len(features)-1))], axis=1)
y_test_real = scaler.inverse_transform(y_real_full)[:,0]

mae = mean_absolute_error(y_test_real, predictions)
print(f"MAE (Sai số trung bình): {mae:.2f} USD")

plt.figure(figsize=(16,8))
plt.plot(y_test_real, label='Giá thực tế', color='blue')
plt.plot(predictions, label='Goldie dự đoán', color='orange', linestyle='--')
plt.title(f'Dự báo giá vàng (Features: Gold, FED, VIX, Sentiment) - MAE: {mae:.2f}')
plt.legend()
plt.savefig(CHART_FILE_PATH)

# --- 8. Dự đoán tương lai (Ngày mai) ---
print("8. Dự đoán giá ngày mai...")
last_60 = scaled_data[-seq_len:]
X_future = np.array([last_60])
pred_scaled = model.predict(X_future)[0][0]
pred_price = scaler.inverse_transform([[pred_scaled, 0, 0, 0]])[0,0]
real_last_price = df['gold'].iloc[-1]

print(f"Giá hôm nay: {real_last_price:.2f} USD | Dự đoán ngày mai: {pred_price:.2f} USD")

# --- 9. Nhật ký dự đoán và Đối soát kết quả ---
print("9. Đang cập nhật Nhật ký dự đoán vào Google Sheet...")
creds, _ = default()
gc = gspread.authorize(creds)

try:
    try:
        sh = gc.open("Goldie_Log")
    except:
        sh = gc.create("Goldie_Log")
        sh.share('', perm_type='anyone', role='writer')
    
    # Dùng Sheet2 để làm Nhật ký (Log)
    try:
        log_sheet = sh.worksheet("Prediction_Log")
    except:
        log_sheet = sh.add_worksheet(title="Prediction_Log", rows="1000", cols="10")
        log_sheet.append_row(["Ngày Dự Báo", "Giá Dự Báo", "Giá Thực Tế", "Chênh Lệch ($)", "Độ Chính Xác (%)"])

    # A. Cập nhật giá thực tế cho dự báo ngày hôm trước
    all_data = log_sheet.get_all_values()
    if len(all_data) > 1: # Nếu đã có dữ liệu
        last_row_idx = len(all_data)
        last_date_str = all_data[-1][0]
        
        # Nếu dòng cuối chưa có giá thực tế
        if all_data[-1][2] == "":
            diff = abs(real_last_price - float(all_data[-1][1]))
            accuracy = max(0, 100 - (diff / real_last_price * 100))
            
            # Cập nhật vào dòng cuối
            log_sheet.update_cell(last_row_idx, 3, f"{real_last_price:.2f}")
            log_sheet.update_cell(last_row_idx, 4, f"{diff:.2f}")
            log_sheet.update_cell(last_row_idx, 5, f"{accuracy:.2f}%")
            print(f"Đã đối soát ngày {last_date_str}: Độ chính xác đạt {accuracy:.2f}%")

    # B. Ghi dòng dự báo mới cho ngày mai
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    log_sheet.append_row([tomorrow, f"{pred_price:.2f}", "", "", ""])
    
    print(f"Đã lưu dự báo cho ngày {tomorrow}")

except Exception as e:
    print(f"Lỗi Nhật ký: {e}")

