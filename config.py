import os

import boto3
from dotenv import load_dotenv
import redis
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
REDIS_URL = os.getenv("REDIS_URL")

# Application
SECRET_KEY = os.getenv("SECRET_KEY")

# OpenAI & Pinecone
API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")

enc = os.getenv("ENCRYPTION_KEY")
s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )
ENCRYPTION_KEY = enc.encode('utf-8')
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=15)
# Проверка, что ключевые переменные установлены
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, DATABASE_URL, SECRET_KEY, S3_BUCKET_NAME, REDIS_URL]):
    raise ValueError("Одна или несколько ключевых переменных окружения не установлены. Проверьте .env файл.")
