import pandas as pd
import numpy as np
from celery import Celery
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import QueuePool
import logging
import time
import hashlib

import crud
import database
from database_analytics_helpers import perform_full_analysis

try:
    from config import REDIS_URL
except ImportError:
    REDIS_URL = "redis://localhost:6379/0"

# Настройки оптимизации
try:
    from config import (
        MAX_DATAFRAME_ROWS,
        MAX_TABLES_DETAILED_ANALYSIS,
        MAX_JOINS_TO_ANALYZE,
        MAX_CHARTS_PER_TABLE,
        MAX_TABLES_TO_VISUALIZE
    )
except ImportError:
    MAX_DATAFRAME_ROWS = 50000
    MAX_TABLES_DETAILED_ANALYSIS = 10
    MAX_JOINS_TO_ANALYZE = 10
    MAX_CHARTS_PER_TABLE = 2
    MAX_TABLES_TO_VISUALIZE = 5

# Инициализация Celery с настройками оптимизации
celery_app = Celery(
    'tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Настройки Celery для производительности
celery_app.conf.update(
    task_acks_late=True,  # Задачи подтверждаются после завершения
    worker_prefetch_multiplier=1,  # Воркер берет только 1 задачу за раз
    task_compression='gzip',  # Сжатие результатов
    result_expires=3600,  # Результаты хранятся 1 час
)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Оптимизирует типы данных DataFrame для снижения потребления памяти
    """
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['object']).columns:
        try:
            num_unique = df[col].nunique()
            num_total = len(df)
            if num_unique / num_total < 0.5:  # Если < 50% уникальных значений
                df[col] = df[col].astype('category')
        except Exception as e:
            logging.warning(f"Не удалось оптимизировать колонку {col}: {e}")

    return df


def smart_sample(df: pd.DataFrame, max_rows: int = MAX_DATAFRAME_ROWS) -> pd.DataFrame:
    """
    Делает интеллектуальную выборку данных для больших таблиц
    """
    if len(df) <= max_rows:
        return df

    logging.info(f"Таблица содержит {len(df)} строк, делаем выборку до {max_rows}")
    return df.sample(n=max_rows, random_state=42)


def get_cache_key(connection_string: str, tables: list) -> str:
    """
    Генерирует ключ для кеширования результатов анализа
    """
    key_string = f"{connection_string}:{''.join(sorted(tables))}"
    return hashlib.md5(key_string.encode()).hexdigest()


@celery_app.task(
    name="run_db_analysis",
    bind=True,
    priority=5,
    rate_limit='10/m',
    time_limit=1800,
    soft_time_limit=1500
)
def run_db_analysis_task(report_id: int, user_id: int, connection_string: str, db_type: str, language: str = "en"):
    start_time = time.time()
    db = database.SessionLocal()
    engine = None

    try:
        logging.warning(f"Celery-задача запущена для отчета ID: {report_id}, пользователь: {user_id}")
        crud.update_report_status(db, report_id, "processing")
        step_start = time.time()
        engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            raise ValueError("В базе данных не найдено таблиц.")

        logging.info(f"[Отчет {report_id}] Найдено таблиц: {len(tables)}")
        logging.info(f"[Отчет {report_id}] Создание подключения: {time.time() - step_start:.2f}s")

        step_start = time.time()
        dataframes = {}
        total_rows = 0
        total_memory = 0

        for table in tables:
            try:
                row_count_query = f"SELECT COUNT(*) FROM {table}"
                row_count = pd.read_sql(row_count_query, con=engine).iloc[0, 0]

                logging.info(f"[Отчет {report_id}] Таблица {table}: {row_count} строк")

                if row_count > 100000:
                    logging.info(f"[Отчет {report_id}] Таблица {table} большая, используем chunked чтение")
                    chunks = []
                    rows_loaded = 0

                    for chunk in pd.read_sql_table(table, con=engine, chunksize=10000):
                        chunks.append(chunk)
                        rows_loaded += len(chunk)
                        if rows_loaded >= MAX_DATAFRAME_ROWS:
                            logging.info(f"[Отчет {report_id}] Достигнут лимит строк для таблицы {table}")
                            break

                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_sql_table(table, con=engine)

                # Оптимизация типов данных
                df = optimize_dtypes(df)

                # Сэмплирование если нужно
                df = smart_sample(df, MAX_DATAFRAME_ROWS)

                dataframes[table] = df

                memory_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
                total_rows += len(df)
                total_memory += memory_usage

                logging.info(
                    f"[Отчет {report_id}] Загружена таблица {table}: "
                    f"{len(df)} строк, {memory_usage:.2f} MB"
                )

            except Exception as e:
                logging.error(f"[Отчет {report_id}] Ошибка при загрузке таблицы {table}: {e}")
                continue

        if not dataframes:
            raise ValueError("Не удалось загрузить ни одной таблицы из базы данных.")

        logging.info(
            f"[Отчет {report_id}] Загрузка данных завершена: "
            f"{len(dataframes)} таблиц, {total_rows} строк, {total_memory:.2f} MB за {time.time() - step_start:.2f}s"
        )

        # Этап 3: Выполнение анализа
        step_start = time.time()

        # Передаем дополнительные параметры для оптимизации
        analysis_results = perform_full_analysis(inspector, dataframes, report_id, language=language)

        logging.info(f"[Отчет {report_id}] Анализ завершен за {time.time() - step_start:.2f}s")

        # Этап 4: Сохранение результатов
        step_start = time.time()
        crud.update_report_results(db, report_id, analysis_results, "completed")
        logging.info(f"[Отчет {report_id}] Сохранение результатов: {time.time() - step_start:.2f}s")

        # Общее время выполнения
        total_duration = time.time() - start_time
        logging.warning(
            f"Celery-задача успешно завершена для отчета ID: {report_id}. "
            f"Общее время: {total_duration:.2f}s ({total_duration / 60:.2f} минут)"
        )

    except Exception as e:
        logging.error(f"Ошибка в Celery-задаче для отчета ID {report_id}: {e}", exc_info=True)

        try:
            crud.update_report_status(db, report_id, "failed", error_message=str(e))
        except Exception as update_error:
            logging.error(f"Не удалось обновить статус отчета: {update_error}")

        # Перебрасываем исключение для Celery retry механизма
        raise

    finally:
        # Закрываем соединения
        db.close()

        if engine is not None:
            engine.dispose()
            logging.info(f"[Отчет {report_id}] Соединения с БД закрыты")


# Опциональная задача для очистки старых кешей
@celery_app.task(name="cleanup_old_caches")
def cleanup_old_caches():
    """
    Периодическая задача для очистки устаревших кешей
    Можно запускать через Celery Beat
    """
    try:
        import redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=False)

        # Удаляем кеши старше 24 часов
        cursor = 0
        deleted_count = 0

        while True:
            cursor, keys = redis_client.scan(cursor, match="analysis:*", count=100)

            for key in keys:
                ttl = redis_client.ttl(key)
                if ttl < 0:  # Ключ без TTL или истекший
                    redis_client.delete(key)
                    deleted_count += 1

            if cursor == 0:
                break

        logging.info(f"Очистка кешей завершена: удалено {deleted_count} записей")

    except Exception as e:
        logging.error(f"Ошибка при очистке кешей: {e}")
