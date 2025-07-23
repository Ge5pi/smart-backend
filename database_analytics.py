# database_analytics.py (обновленный)
import json
import uuid
from datetime import timedelta
from typing import Dict, Any
import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session
from openai import OpenAI

import auth
import config  # Импортируем config для redis_client и API_KEY
import crud
import database
import models

# Используем из config
from config import redis_client, API_KEY

database_router = APIRouter(prefix="/analytics/database")
client = OpenAI(api_key=API_KEY)


def save_dataframes_to_redis(session_id: str, dataframes: dict[str, pd.DataFrame]):
    """Saves DataFrames to Redis with proper serialization."""
    serialized_dfs = {}
    for table, df in dataframes.items():
        df_dict = df.to_dict(orient='records')
        for row in df_dict:
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    row[key] = value.isoformat()
        serialized_dfs[table] = df_dict

    redis_client.setex(session_id, timedelta(hours=2), json.dumps({"dataframes": serialized_dfs}))


def load_dataframes_from_redis(session_id: str) -> Dict[str, pd.DataFrame]:
    """Загружает DataFrame из Redis."""
    data = redis_client.get(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Сессия не найдена или истекла.")
    session_data = json.loads(data)
    return {table: pd.DataFrame(df_data) for table, df_data in session_data["dataframes"].items()}


async def perform_analysis(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Анализирует данные с помощью Pandas и GPT API для выявления паттернов и корреляций."""
    insights = {}
    correlations = {}
    for table, df in dataframes.items():
        # Вычисление корреляций с помощью Pandas
        corr = df.corr(numeric_only=True).to_dict()
        correlations[table] = corr
        stats = df.describe().to_json()
        # Использование GPT для интерпретации
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
    """Генерирует отчет на основе анализа и сохраняет его в базе данных."""
    analysis_results = await perform_analysis(dataframes)
    report = models.Report(user_id=user_id, status="completed", results=analysis_results)
    db.add(report)
    db.commit()
    return report.id


@database_router.post("/analyze")
async def analyze_database(
        connectionString: str = Form(...),
        dbType: str = Form(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Анализирует базу данных на основе строки подключения."""
    if dbType not in ['postgres', 'sqlserver']:
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип базы данных.")

    try:
        if dbType == 'postgres':
            engine = create_engine(connectionString)
        else:  # sqlserver
            engine = create_engine(connectionString, echo=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка подключения к базе данных: {str(e)}")

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if not tables:
            raise HTTPException(status_code=404, detail="Таблицы в базе данных не найдены.")

        dataframes = {}
        for table in tables:
            df = pd.read_sql_table(table, con=engine)
            dataframes[table] = df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при извлечении таблиц: {str(e)}")
    finally:
        engine.dispose()

    session_id = str(uuid.uuid4())
    save_dataframes_to_redis(session_id, dataframes)
    report_id = await generate_report(session_id, dataframes, current_user.id, db)
    return {"report_id": report_id, "message": "Анализ базы данных запущен."}
