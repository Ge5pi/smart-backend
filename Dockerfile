FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fontconfig \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Обновление кэша шрифтов
RUN fc-cache -fv

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё остальное
COPY . .

# Скрипт запуска
RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
