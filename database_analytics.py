# database_analytics.py
import json
import uuid
from datetime import timedelta
from typing import Dict, Any
import pandas as pd
import numpy as np
import io
import boto3
from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session
from openai import OpenAI
import logging
import json
import auth
import config
import crud
import database
import models

# Используем из config (предполагаем S3-клиент и Redis)
from config import redis_client, API_KEY, S3_BUCKET_NAME, s3_client

database_router = APIRouter(prefix="/analytics/database")
client = OpenAI(api_key=API_KEY)


def save_dataframes_to_s3(session_id: str, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """Сохраняет DataFrames в S3 как Parquet и возвращает ключи."""
    s3_keys = {}
    for table, df in dataframes.items():
        # Обрабатываем NaN и Timestamp перед сохранением
        df = df.replace({np.nan: None})
        df = df.applymap(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)

        # Сохраняем как Parquet в буфер
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        key = f"sessions/{session_id}/{table}.parquet"
        s3_client.put_object(Bucket=config.S3_BUCKET_NAME, Key=key, Body=buffer, ContentType='application/octet-stream')
        s3_keys[table] = key
    return s3_keys


def load_dataframes_from_s3(s3_keys: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """Загружает DataFrames из S3."""
    dataframes = {}
    for table, key in s3_keys.items():
        obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        buffer = io.BytesIO(obj['Body'].read())
        df = pd.read_parquet(buffer)
        dataframes[table] = df
    return dataframes


async def perform_analysis(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    insights = {}
    correlations = {}
    for table, df in dataframes.items():
        corr = df.corr().replace({np.nan: None}).to_dict()
        correlations[table] = corr
        stats = df.describe().replace({np.nan: None}).to_json()
        prompt = (
            f"Анализируй данные таблицы '{table}'. Вот статистика: {stats}. "
            f"Корреляции: {json.dumps(corr)}. Выяви скрытые паттерны, инсайты и корреляции. "
            f"Ответ должен быть на русском языке, кратким и структурированным."
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        insights[table] = response.choices[0].message.content
    return {"insights": insights, "correlations": correlations}


async def generate_report(session_id: str, dataframes: Dict[str, pd.DataFrame], user_id: int, db: Session) -> int:
    analysis_results = await perform_analysis(dataframes)

    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    cleaned_results = clean_nan(analysis_results)
    report = models.Report(user_id=user_id, status="completed", results=cleaned_results)
    db.add(report)
    db.commit()
    db.refresh(report) # <-- ДОБАВЛЕНО: Обновляем объект из БД, чтобы получить id
    logging.warning(f"Создан отчет с ID: {report.id}") # <-- ДОБАВЛЕНО: Логируем ID
    return report.id


@database_router.post("/analyze")
async def analyze_database(
        connectionString: str = Form(...),
        dbType: str = Form(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    if dbType not in ['postgres', 'sqlserver']:
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип базы данных.")

    try:
        engine = create_engine(connectionString) if dbType == 'postgres' else create_engine(connectionString, echo=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка подключения: {str(e)}")

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if not tables:
            raise HTTPException(status_code=404, detail="Таблицы не найдены.")

        dataframes = {table: pd.read_sql_table(table, con=engine) for table in tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка извлечения таблиц: {str(e)}")
    finally:
        engine.dispose()

    session_id = str(uuid.uuid4())
    s3_keys = save_dataframes_to_s3(session_id, dataframes)  # Сохраняем в S3
    redis_client.setex(session_id, timedelta(hours=2),
                       json.dumps({"s3_keys": s3_keys}))  # В Redis только ключи (малый размер)
    report_id = await generate_report(session_id, dataframes, current_user.id, db)
    return {"report_id": report_id, "message": "Анализ запущен."}


@database_router.get("/reports/{report_id}")
async def get_report_details(
    report_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    """
    Retrieves a specific report by its ID.
    """
    logging.warning(f"Поиск отчета с ID: {report_id} для пользователя {current_user.id}") # <-- ДОБАВЛЕНО
    report = crud.get_report_by_id(db, report_id=report_id)
    logging.warning(f"Результат из БД: {report}") # <-- ДОБАВЛЕНО

    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден.")

    if report.user_id != current_user.id:
        logging.warning(f"Доступ запрещен для отчета {report.id}. Владелец: {report.user_id}, запрашивает: {current_user.id}") # <-- ДОБАВЛЕНО
        raise HTTPException(status_code=403, detail="Недостаточно прав для просмотра этого отчета.")

    return report
