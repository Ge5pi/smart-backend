from celery import Celery
from sqlalchemy import create_engine, inspect
import logging

# Важно импортировать все необходимые модули вашего проекта
import crud
import database
import models
from database_analytics import perform_full_analysis  # Импортируем основную логику

# Предполагается, что в вашем config.py есть URL для Redis
# Например: REDIS_URL = "redis://localhost:6379/0"
try:
    from config import REDIS_URL
except ImportError:
    REDIS_URL = "redis://localhost:6379/0"  # Fallback для примера

# Настраиваем Celery
celery_app = Celery(
    'tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)


@celery_app.task(name="run_db_analysis")
def run_db_analysis_task(report_id: int, user_id: int, connection_string: str, db_type: str):
    """
    Фоновая задача Celery для полного анализа базы данных.
    """
    db = database.SessionLocal()
    try:
        logging.warning(f"Celery-задача запущена для отчета ID: {report_id}")

        # 1. Устанавливаем статус "в обработке"
        crud.update_report_status(db, report_id, "processing")

        # 2. Выполняем тяжелую работу (как раньше было в эндпоинте)
        engine = create_engine(connection_string)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if not tables:
            raise ValueError("В базе данных не найдено таблиц.")

        dataframes = {}
        for table in tables:
            dataframes[table] = pd.read_sql_table(table, con=engine)

        # 3. Запускаем основной процесс анализа (текст + графики)
        analysis_results = perform_full_analysis(inspector, dataframes, report_id)

        # 4. Обновляем отчет с результатами и статусом "завершено"
        crud.update_report_results(db, report_id, analysis_results, "completed")
        logging.warning(f"Celery-задача успешно завершена для отчета ID: {report_id}")

    except Exception as e:
        logging.error(f"Ошибка в Celery-задаче для отчета ID {report_id}: {e}", exc_info=True)
        # 5. В случае ошибки обновляем статус
        crud.update_report_status(db, report_id, "failed", error_message=str(e))
    finally:
        db.close()
        if 'engine' in locals() and engine:
            engine.dispose()