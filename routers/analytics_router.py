# analytics_router.py - обновленный с DataFrame поддержкой
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from celery.result import AsyncResult
from tasks import (
    generate_dataframe_report,
    quick_dataframe_analysis,
    comprehensive_dataframe_analysis,
    generate_advanced_report  # Для обратной совместимости
)
from celery_worker import celery_app
import crud
import schemas
import models
import auth
import database
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analytics",
    tags=["DataFrame Analytics Engine"],
    dependencies=[Depends(auth.get_current_active_user)]
)


# =============== ПОДКЛЮЧЕНИЯ К БД ===============

@router.post("/connections/", response_model=schemas.DatabaseConnectionInfo)
def add_database_connection(
        conn_details: schemas.DatabaseConnectionCreate,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Добавляет новое подключение к базе данных с проверкой."""
    try:
        # Проверяем подключение
        engine = create_engine(conn_details.connection_string, connect_args={'connect_timeout': 10})
        connection = engine.connect()
        connection.close()

        # Создаем запись в БД
        db_conn = crud.create_db_connection(db, user_id=current_user.id, conn_details=conn_details)
        logger.info(f"Пользователь {current_user.id} добавил новое подключение {db_conn.id}")
        return db_conn

    except SQLAlchemyError as e:
        logger.error(f"Ошибка подключения к БД: {e}")
        raise HTTPException(status_code=400, detail=f"Неверная строка подключения: {e}")


@router.get("/connections/", response_model=List[schemas.DatabaseConnectionInfo])
def get_user_connections(
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает все подключения пользователя."""
    connections = crud.get_db_connections_by_user(db, user_id=current_user.id)
    logger.info(f"Пользователь {current_user.id} запросил список подключений: {len(connections)} найдено")
    return connections


# =============== ОСНОВНОЙ DATAFRAME АНАЛИЗ ===============

@router.post("/reports/generate-dataframe/{connection_id}", response_model=schemas.ReportInfo)
def generate_dataframe_analytics_report(
        connection_id: int,
        max_questions: Optional[int] = 15,
        analysis_type: Optional[str] = "standard",  # standard, quick, comprehensive
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """
    Генерирует отчет на основе DataFrame без использования SQL

    Args:
        connection_id: ID подключения к БД
        max_questions: Максимальное количество вопросов для анализа
        analysis_type: Тип анализа - standard, quick, comprehensive
    """
    # Проверяем существование подключения
    db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
    if not db_conn:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    # Создаем запись отчета
    report_record = crud.create_report(
        db=db,
        user_id=current_user.id,
        connection_id=connection_id
    )

    # Выбираем тип анализа и запускаем соответствующую задачу
    if analysis_type == "quick":
        logger.info(f"Запуск быстрого DataFrame-анализа для пользователя {current_user.id}")
        task = quick_dataframe_analysis.delay(connection_id, current_user.id, report_record.id)
    elif analysis_type == "comprehensive":
        logger.info(f"Запуск комплексного DataFrame-анализа для пользователя {current_user.id}")
        task = comprehensive_dataframe_analysis.delay(connection_id, current_user.id, report_record.id)
    else:  # standard
        logger.info(f"Запуск стандартного DataFrame-анализа для пользователя {current_user.id}")
        task = generate_dataframe_report.delay(connection_id, current_user.id, report_record.id, max_questions)

    # Обновляем запись с task_id
    report_record.task_id = task.id
    db.commit()
    db.refresh(report_record)

    logger.info(f"DataFrame-отчет {report_record.id} создан с task_id: {task.id}")
    return report_record


# =============== ОБНОВЛЕННЫЙ ОСНОВНОЙ ЭНДПОИНТ ===============

@router.post("/reports/generate/{connection_id}", response_model=schemas.ReportInfo)
def generate_analytics_report(
        connection_id: int,
        max_questions: Optional[int] = 15,
        use_dataframe: Optional[bool] = True,  # По умолчанию используем DataFrame
        analysis_type: Optional[str] = "standard",
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """
    Генерирует аналитический отчет (теперь по умолчанию DataFrame)

    Args:
        connection_id: ID подключения к БД
        max_questions: Максимальное количество вопросов
        use_dataframe: Использовать DataFrame подход (по умолчанию True)
        analysis_type: Тип анализа
    """
    if use_dataframe:
        # Используем новый DataFrame подход
        return generate_dataframe_analytics_report(
            connection_id=connection_id,
            max_questions=max_questions,
            analysis_type=analysis_type,
            db=db,
            current_user=current_user
        )
    else:
        # Legacy SQL подход (для обратной совместимости)
        logger.warning(f"Используется legacy SQL подход для пользователя {current_user.id}")

        db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
        if not db_conn:
            raise HTTPException(status_code=404, detail="Подключение не найдено")

        report_record = crud.create_report(
            db=db,
            user_id=current_user.id,
            connection_id=connection_id
        )

        task = generate_advanced_report.delay(connection_id, current_user.id, report_record.id)

        report_record.task_id = task.id
        db.commit()
        db.refresh(report_record)

        return report_record


# =============== LEGACY ЭНДПОИНТЫ ===============

@router.post("/reports/generate-legacy/{connection_id}", response_model=schemas.ReportInfo)
def generate_legacy_report(
        connection_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Legacy эндпоинт для старых клиентов (использует SQL подход)."""
    return generate_analytics_report(
        connection_id=connection_id,
        max_questions=12,
        use_dataframe=False,
        analysis_type="legacy",
        db=db,
        current_user=current_user
    )


# =============== СТАТУС И ПОЛУЧЕНИЕ ОТЧЕТОВ ===============

@router.get("/reports/status/{task_id}")
def get_report_status(task_id: str):
    """Получает расширенный статус задачи анализа."""
    task_result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": task_result.status,
        "info": task_result.info
    }

    if task_result.info and isinstance(task_result.info, dict):
        response.update({
            "progress": task_result.info.get("progress", "Неизвестно"),
            "stage": task_result.info.get("stage", "unknown"),
            "progress_percentage": task_result.info.get("progress_percentage", 0),
            "method": task_result.info.get("method", "unknown")
        })

        if task_result.info.get("error"):
            response["error"] = task_result.info["error"]

    logger.info(f"Запрошен статус задачи {task_id}: {task_result.status}")
    return response


@router.get("/reports/list/", response_model=List[schemas.ReportInfo])
def get_user_reports(
        limit: Optional[int] = 10,
        offset: Optional[int] = 0,
        status_filter: Optional[str] = None,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает список отчетов пользователя с фильтрацией."""
    reports = crud.get_user_reports(
        db,
        user_id=current_user.id,
        limit=limit,
        offset=offset,
        status_filter=status_filter
    )

    logger.info(f"Пользователь {current_user.id} запросил список отчетов: {len(reports)} найдено")
    return reports


@router.get("/reports/{report_id}", response_model=schemas.Report)
def get_report_by_id(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает готовый отчет по ID с полной информацией."""
    report = crud.get_report_by_id(db, report_id=report_id, user_id=current_user.id)
    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")

    logger.info(f"Пользователь {current_user.id} получил данные отчета {report_id}")

    # Обеспечиваем правильный формат results
    if isinstance(report.results, str):
        try:
            report.results = json.loads(report.results)
        except json.JSONDecodeError:
            report.results = {"error": "Malformed report data in database."}

    return report


# =============== ОБРАТНАЯ СВЯЗЬ ===============

@router.post("/reports/feedback/{report_id}", response_model=schemas.FeedbackResponse)
def submit_report_feedback(
        report_id: int,
        feedback_data: schemas.FeedbackCreate,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Отправляет обратную связь по отчету."""
    try:
        feedback_record = crud.create_report_feedback(
            db,
            report_id=report_id,
            user_id=current_user.id,
            feedback_data=feedback_data.model_dump()
        )

        logger.info(f"Получена обратная связь для отчета {report_id} от пользователя {current_user.id}")
        return feedback_record

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# =============== ДИАГНОСТИКА И МОНИТОРИНГ ===============

@router.get("/health-check/{connection_id}")
def check_database_health(
        connection_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Проверяет состояние подключения к базе данных."""
    db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
    if not db_conn:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    try:
        connection_string = crud.get_decrypted_connection_string(db, connection_id, current_user.id)
        engine = create_engine(connection_string, connect_args={'connect_timeout': 10})

        # Создаем простую проверку здоровья для DataFrame
        from services.dataframe_manager import DataFrameManager

        df_manager = DataFrameManager(engine)

        # Пробуем загрузить таблицы (с ограничением)
        with engine.connect() as conn:
            from sqlalchemy import inspect, text
            inspector = inspect(engine)
            tables = [t for t in inspector.get_table_names() if t != 'alembic_version']

            total_rows = 0
            tables_with_data = 0

            for table in tables[:5]:  # Проверяем только первые 5 таблиц
                try:
                    count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
                    total_rows += count
                    if count > 0:
                        tables_with_data += 1
                except:
                    continue

            health_check = {
                "connection": True,
                "has_data": total_rows > 0,
                "total_tables": len(tables),
                "tables_with_data": tables_with_data,
                "total_rows": total_rows,
                "dataframe_compatible": True,
                "estimated_memory_mb": round(total_rows * len(tables) * 0.001, 2)  # Грубая оценка
            }

        return health_check

    except Exception as e:
        logger.error(f"Ошибка проверки здоровья БД {connection_id}: {e}")
        return {
            "connection": False,
            "has_data": False,
            "error": str(e),
            "dataframe_compatible": False
        }


@router.get("/system/stats")
def get_system_statistics(
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает статистику системы анализа."""
    i = celery_app.control.inspect()
    active_tasks = i.active()
    scheduled_tasks = i.scheduled()

    active_count = sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0
    scheduled_count = sum(len(tasks) for tasks in scheduled_tasks.values()) if scheduled_tasks else 0

    stats = {
        "active_tasks": active_count,
        "scheduled_tasks": scheduled_count,
        "user_id": current_user.id,
        "available_analysis_types": ["standard", "quick", "comprehensive", "legacy"],
        "default_method": "dataframe",
        "max_questions_limit": 50,
        "supported_features": [
            "dataframe_analysis",
            "relationship_detection",
            "anomaly_detection",
            "correlation_analysis",
            "trend_analysis",
            "memory_optimization",
            "automatic_chart_generation"
        ],
        "memory_limits": {
            "max_table_rows": 100000,
            "max_total_memory_mb": 2048
        }
    }

    return stats


# =============== ДОПОЛНИТЕЛЬНЫЕ DATAFRAME УТИЛИТЫ ===============

@router.get("/dataframe/preview/{connection_id}")
def preview_dataframe_tables(
        connection_id: int,
        max_rows_per_table: Optional[int] = 5,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Быстрый preview таблиц в DataFrame формате."""
    db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
    if not db_conn:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    try:
        connection_string = crud.get_decrypted_connection_string(db, connection_id, current_user.id)
        engine = create_engine(connection_string, connect_args={'connect_timeout': 10})

        from services.dataframe_manager import DataFrameManager

        df_manager = DataFrameManager(engine)

        # Загружаем только preview
        preview_tables = {}
        from sqlalchemy import inspect
        inspector = inspect(engine)

        for table_name in inspector.get_table_names()[:10]:  # Максимум 10 таблиц
            if table_name == 'alembic_version':
                continue

            try:
                import pandas as pd
                df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT {max_rows_per_table}", engine)
                if not df.empty:
                    preview_tables[table_name] = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': df.columns.tolist(),
                        'sample_data': df.head(max_rows_per_table).to_dict('records')
                    }
            except Exception as e:
                logger.error(f"Ошибка preview таблицы {table_name}: {e}")
                continue

        return {
            "connection_id": connection_id,
            "tables_preview": preview_tables,
            "total_tables_found": len(preview_tables),
            "max_rows_per_table": max_rows_per_table
        }

    except Exception as e:
        logger.error(f"Ошибка DataFrame preview: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения preview: {e}")


@router.get("/reports/debug/{report_id}")
def debug_report(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Отладочный эндпоинт для проверки отчета"""
    report = crud.get_report_by_id(db, report_id, current_user.id)
    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")

    return {
        "id": report.id,
        "status": report.status,
        "created_at": report.created_at,
        "has_results": bool(report.results),
        "results_keys": list(report.results.keys()) if report.results else [],
        "task_id": report.task_id
    }
