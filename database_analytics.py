import uuid
from typing import Dict, Any, Set, Tuple
import pandas as pd
import numpy as np
import io
from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Inspector
from sqlalchemy.orm import Session
from openai import OpenAI
import logging
import json
import auth
import config
import crud
import database
import models
import schemas
from config import API_KEY

database_router = APIRouter(prefix="/analytics/database")
client = OpenAI(api_key=API_KEY)


def analyze_single_table(table_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Анализирует отдельную таблицу и возвращает инсайты и корреляции."""
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr().replace({np.nan: None}).to_dict() if not numeric_df.empty else {}
    stats = df.describe(include='all').replace({np.nan: None}).to_json()

    prompt = (
        f"Проанализируй данные из таблицы '{table_name}'. "
        f"Вот описательная статистика: {stats}. "
        f"Вот матрица корреляций для числовых полей: {json.dumps(corr)}. "
        "Твоя задача — выявить ключевые инсайты, скрытые закономерности и аномалии в данных этой таблицы. "
        "Будь кратким, структурированным и пиши на русском языке."
    )

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    insight = response.choices[0].message.content
    return {"insight": insight, "correlations": corr}


def analyze_joins(inspector: Inspector, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Находит внешние ключи, объединяет таблицы и анализирует результат.
    """
    joint_insights = {}
    analyzed_pairs: Set[Tuple[str, str]] = set()
    all_tables = list(dataframes.keys())

    for table_name in all_tables:
        try:
            foreign_keys = inspector.get_foreign_keys(table_name)
        except Exception as e:
            logging.warning(f"Не удалось получить внешние ключи для таблицы {table_name}: {e}")
            continue

        for fk in foreign_keys:
            left_table = table_name
            right_table = fk['referred_table']

            # Сортируем имена таблиц, чтобы избежать дублирования (user, post) и (post, user)
            pair = tuple(sorted((left_table, right_table)))
            if pair in analyzed_pairs:
                continue

            analyzed_pairs.add(pair)

            df_left = dataframes[left_table]
            df_right = dataframes.get(right_table)

            if df_right is None:
                continue

            left_on = fk['constrained_columns']
            right_on = fk['referred_columns']

            try:
                # Объединяем таблицы, добавляя суффиксы, чтобы избежать конфликта имен столбцов
                merged_df = pd.merge(
                    df_left, df_right,
                    left_on=left_on,
                    right_on=right_on,
                    suffixes=(f'_{left_table}', f'_{right_table}')
                )

                if merged_df.empty:
                    continue

                join_key = f"{left_table} 🔗 {right_table}"

                # Анализируем объединенный DataFrame
                analysis_result = analyze_single_table(join_key, merged_df)

                # Корректируем промпт для GPT, чтобы он фокусировался на межтабличных связях
                stats = merged_df.describe(include='all').replace({np.nan: None}).to_json()
                corr = analysis_result["correlations"]

                prompt = (
                    f"Проанализируй СВЯЗЬ между таблицами '{left_table}' и '{right_table}', которые были объединены по ключам "
                    f"({left_table}.{','.join(left_on)} = {right_table}.{','.join(right_on)}). "
                    f"Вот статистика объединенных данных: {stats}. "
                    f"Вот матрица корреляций: {json.dumps(corr)}. "
                    f"Сосредоточься на поиске инсайтов, которые возникают именно из-за связи двух таблиц. "
                    f"Например, как атрибуты из одной таблицы влияют на атрибуты в другой? "
                    "Ответ дай на русском языке, кратко и по делу."
                )

                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5
                )

                analysis_result["insight"] = response.choices[0].message.content
                joint_insights[join_key] = analysis_result

            except Exception as e:
                logging.error(f"Ошибка при объединении и анализе таблиц {left_table} и {right_table}: {e}")

    return joint_insights


async def perform_full_analysis(inspector: Inspector, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Выполняет полный анализ: по каждой таблице и по их связям."""

    # 1. Анализ по отдельным таблицам
    single_table_analysis = {}
    for table, df in dataframes.items():
        single_table_analysis[table] = analyze_single_table(table, df)

    # 2. Анализ межтабличных связей
    joint_table_analysis = analyze_joins(inspector, dataframes)

    return {
        "single_table_insights": single_table_analysis,
        "joint_table_insights": joint_table_analysis
    }


async def generate_report(session_id: str, inspector: Inspector, dataframes: Dict[str, pd.DataFrame], user_id: int,
                          db: Session) -> int:
    # Запускаем новый, улучшенный процесс анализа
    analysis_results = await perform_full_analysis(inspector, dataframes)

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
    db.refresh(report)
    logging.warning(f"Создан отчет с ID: {report.id}")
    return report.id


@database_router.post("/analyze")
async def analyze_database(
        connectionString: str = Form(...),
        dbType: str = Form(...),
        alias: str = Form(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    if dbType not in ['postgres', 'sqlserver']:
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип базы данных.")

    crud.create_database_connection(db, user_id=current_user.id, connection_string=connectionString, db_type=dbType,
                                    alias=alias)

    engine = None
    try:
        engine = create_engine(connectionString)
        inspector = inspect(engine)

        tables = inspector.get_table_names()
        if not tables:
            raise HTTPException(status_code=404, detail="В базе данных не найдено таблиц.")

        dataframes = {}
        for table in tables:
            try:
                dataframes[table] = pd.read_sql_table(table, con=engine)
            except Exception as e:
                logging.warning(f"Не удалось прочитать таблицу {table}: {e}. Пропускаем.")

        if not dataframes:
            raise HTTPException(status_code=500, detail="Не удалось прочитать ни одну таблицу из базы данных.")

        session_id = str(uuid.uuid4())
        # Передаем inspector и dataframes для полного анализа
        report_id = await generate_report(session_id, inspector, dataframes, current_user.id, db)
        return {"report_id": report_id, "message": "Анализ успешно завершен."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Произошла критическая ошибка при анализе: {str(e)}")
    finally:
        if engine:
            engine.dispose()


# ... (остальной код без изменений: get_user_connections, get_report_details) ...
# Копипаст остального кода из database_analytics.py
@database_router.get("/connections", response_model=list[schemas.DatabaseConnection])
async def get_user_connections(
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """
    Возвращает список сохраненных подключений для текущего пользователя.
    """
    return crud.get_database_connections_by_user_id(db, user_id=current_user.id)


@database_router.get("/reports/{report_id}")
async def get_report_details(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """
    Получает конкретный отчет по его ID.
    """
    report = crud.get_report_by_id(db, report_id=report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден.")

    if report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Недостаточно прав для просмотра этого отчета.")

    return report