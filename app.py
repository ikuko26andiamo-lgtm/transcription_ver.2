FROM python:3.11-slim

WORKDIR /app

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    libmecab-dev \
    mecab-ipadic-utf8 \
    pkg-config \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libswscale-dev \
    libavutil-dev \
    && rm -rf /var/lib/apt/lists/*

# pipアップグレード
RUN pip install --no-cache-dir --upgrade pip

# 依存関係のコピーとインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 全ファイルのコピー
COPY . .

# ポート設定
EXPOSE 10000

# 起動
CMD ["streamlit", "run", "app.py", \
     "--server.port", "10000", \
     "--server.address", "0.0.0.0", \
     "--browser.gatherUsageStats", "false"]
