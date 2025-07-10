import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

# S3
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Database
DATABASE_URL = os.getenv("DATABASE_URL")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Application
SECRET_KEY = os.getenv("SECRET_KEY")

# OpenAI & Pinecone
API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")


# Проверка, что ключевые переменные установлены
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, DATABASE_URL, SECRET_KEY, S3_BUCKET_NAME, REDIS_HOST]):
    raise ValueError("Одна или несколько ключевых переменных окружения не установлены. Проверьте .env файл.")