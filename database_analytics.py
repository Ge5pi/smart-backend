from typing import Dict, Any, Set, Tuple, List
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
import matplotlib.pyplot as plt
import seaborn as sns
import auth
import config
import crud
import database
import models
import schemas
from config import API_KEY

database_router = APIRouter(prefix="/analytics/database")
client = OpenAI(api_key=API_KEY)


# --- Функции analyze_single_table и analyze_joins остаются без изменений ---
def analyze_single_table(table_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr().replace({np.nan: None}).to_dict() if not numeric_df.empty else {}
    stats = df.describe(include='all').replace({np.nan: None}).to_json()

    prompt = (
        f"Проанализируй данные из таблицы '{table_name}'. "
        f"Вот описательная статистика: {stats}. "
        f"Вот матрица корреляций для числовых полей: {json.dumps(corr)}. "
        "Твоя задача — выявить ключевые инсайты, скрытые закономерности и аномалии в данных этой таблицы. "
        "Будь кратким, структурированным и пиши на русском языке, используя Markdown для выделения (`**термин**`)."
    )

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    insight = response.choices[0].message.content
    return {"insight": insight, "correlations": corr}


def analyze_joins(inspector: Inspector, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
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
                merged_df = pd.merge(
                    df_left, df_right,
                    left_on=left_on,
                    right_on=right_on,
                    suffixes=(f'_{left_table}', f'_{right_table}')
                )

                if merged_df.empty:
                    continue

                join_key = f"{left_table} 🔗 {right_table}"
                analysis_result = analyze_single_table(join_key, merged_df)
                stats = merged_df.describe(include='all').replace({np.nan: None}).to_json()
                corr = analysis_result["correlations"]

                prompt = (
                    f"Проанализируй СВЯЗЬ между таблицами '{left_table}' и '{right_table}', которые были объединены по ключам "
                    f"({left_table}.{','.join(left_on)} = {right_table}.{','.join(right_on)}). "
                    f"Вот статистика объединенных данных: {stats}. "
                    f"Вот матрица корреляций: {json.dumps(corr)}. "
                    f"Сосредоточься на поиске инсайтов, которые возникают именно из-за связи двух таблиц. "
                    f"Например, как атрибуты из одной таблицы влияют на атрибуты в другой? "
                    "Ответ дай на русском языке, кратко и по делу, используя Markdown для выделения (`**термин**`)."
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


# ОБНОВЛЕННАЯ ФУНКЦИЯ для генерации графиков с pre-signed URLs
def generate_visualizations(
        dataframes: Dict[str, pd.DataFrame], report_id: int
) -> Dict[str, List[str]]:
    visualizations = {}
    sns.set_theme(style="whitegrid")

    for name, df in dataframes.items():
        if df.empty:
            continue

        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        df_info = df.dtypes.to_string()
        prompt = (
            f"Для DataFrame с названием '{name}' и следующей структурой столбцов:\n{df_info}\n\n"
            "Предложи до 2 наиболее подходящих визуализаций для анализа этих данных. "
            "Ответ предоставь в виде JSON-массива. Каждый объект в массиве должен содержать: "
            "'chart_type' (тип графика: 'hist', 'bar', 'scatter', 'pie'), "
            "'columns' (список столбцов для использования), и 'title' (название графика на русском). "
            "Для 'bar' первый столбец - категориальный, второй - числовой. Для 'hist' - один числовой столбец. "
            "Для 'scatter' - два числовых столбца. Для 'pie' - один категориальный столбец (до 10 уникальных значений). "
            "Выбирай столбцы с умом. Не предлагай scatter если нет двух числовых колонок. Не предлагай pie для колонок с большим количеством уникальных значений."
            "Возвращай только JSON."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            chart_ideas = json.loads(response.choices[0].message.content)

            chart_urls = []
            for i, idea in enumerate(chart_ideas):
                plt.figure(figsize=(10, 6))

                chart_type = idea.get("chart_type")
                columns = idea.get("columns", [])
                title = idea.get("title", "Сгенерированный график")

                try:
                    if chart_type == 'hist' and len(columns) == 1 and pd.api.types.is_numeric_dtype(df[columns[0]]):
                        sns.histplot(df, x=columns[0], kde=True)
                    elif chart_type == 'bar' and len(columns) == 2:
                        top_15 = df.groupby(columns[0])[columns[1]].sum().nlargest(15)
                        sns.barplot(x=top_15.index, y=top_15.values)
                        plt.xticks(rotation=45, ha='right')
                    elif chart_type == 'scatter' and len(columns) == 2 and all(
                            pd.api.types.is_numeric_dtype(df[c]) for c in columns):
                        sns.scatterplot(df, x=columns[0], y=columns[1])
                    elif chart_type == 'pie' and len(columns) == 1 and df[columns[0]].nunique() <= 10:
                        df[columns[0]].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False)
                        plt.ylabel('')
                    else:
                        continue

                    plt.title(title)
                    plt.tight_layout()

                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight')
                    buffer.seek(0)

                    s3_key = f"charts/{report_id}/{safe_name}_{i}.png"

                    # ИЗМЕНЕНИЕ: Убираем ACL и загружаем приватный объект
                    config.s3_client.put_object(
                        Bucket=config.S3_BUCKET_NAME,
                        Key=s3_key,
                        Body=buffer,
                        ContentType='image/png'
                    )

                    # ИЗМЕНЕНИЕ: Генерируем временную ссылку на 1 час
                    presigned_url = config.s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': config.S3_BUCKET_NAME, 'Key': s3_key},
                        ExpiresIn=3600  # Ссылка действительна 1 час
                    )
                    chart_urls.append(presigned_url)

                except Exception as e:
                    logging.error(f"Не удалось создать график '{title}': {e}", exc_info=True)
                finally:
                    plt.close()

            if chart_urls:
                visualizations[name] = chart_urls

        except Exception as e:
            logging.error(f"Ошибка при генерации идей для графиков для '{name}': {e}")

    return visualizations


# --- Остальные функции (perform_full_analysis, generate_report, analyze_database, etc.) остаются без изменений ---
async def perform_full_analysis(
        inspector: Inspector, dataframes: Dict[str, pd.DataFrame], report_id: int
) -> Dict[str, Any]:
    single_table_analysis = {}
    for table, df in dataframes.items():
        single_table_analysis[table] = analyze_single_table(table, df)

    joint_table_analysis = analyze_joins(inspector, dataframes)

    visualizations = generate_visualizations(dataframes.copy(), report_id)

    return {
        "single_table_insights": single_table_analysis,
        "joint_table_insights": joint_table_analysis,
        "visualizations": visualizations
    }


async def generate_report(user_id: int, inspector: Inspector, dataframes: Dict[str, pd.DataFrame], db: Session) -> int:
    report = models.Report(user_id=user_id, status="pending")
    db.add(report)
    db.commit()
    db.refresh(report)

    report_id = report.id

    analysis_results = await perform_full_analysis(inspector, dataframes, report_id)

    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    cleaned_results = clean_nan(analysis_results)

    report.status = "completed"
    report.results = cleaned_results
    db.commit()

    logging.warning(f"Создан и обновлен отчет с ID: {report.id}")
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

        report_id = await generate_report(current_user.id, inspector, dataframes, db)
        return {"report_id": report_id, "message": "Анализ успешно завершен."}

    except Exception as e:
        logging.error("Критическая ошибка при анализе", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Произошла критическая ошибка при анализе: {str(e)}")
    finally:
        if engine:
            engine.dispose()


@database_router.get("/connections", response_model=list[schemas.DatabaseConnection])
async def get_user_connections(
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    return crud.get_database_connections_by_user_id(db, user_id=current_user.id)


@database_router.get("/reports/{report_id}")
async def get_report_details(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    report = crud.get_report_by_id(db, report_id=report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден.")

    if report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Недостаточно прав для просмотра этого отчета.")

    return report