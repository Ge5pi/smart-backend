FROM python:3.11-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё остальное
COPY . .

# Скрипт запуска
RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
