# routers/analytics_router.py

import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from celery.result import AsyncResult
from tasks import (
    generate_dataframe_report,
    quick_dataframe_analysis,
    comprehensive_dataframe_analysis,
    generate_advanced_report  # Legacy
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
    tags=["SmartGPT DataFrame Analytics"],
    dependencies=[Depends(auth.get_current_active_user)]
)


# =============== ПОДКЛЮЧЕНИЯ К БД ===============

@router.post("/connections/", response_model=schemas.DatabaseConnectionInfo)
def add_database_connection(
        conn_details: schemas.DatabaseConnectionCreate,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Добавляет новое подключение к базе данных с проверкой SmartGPT совместимости."""
    try:
        # Проверяем подключение
        engine = create_engine(conn_details.connection_string, connect_args={'connect_timeout': 10})

        # Тестируем подключение
        with engine.connect() as conn:
            from sqlalchemy import inspect, text
            inspector = inspect(engine)
            tables = inspector.get_table_names()

            # Подсчитываем базовую статистику для оценки SmartGPT потенциала
            total_rows = 0
            for table in tables[:5]:  # Проверяем первые 5 таблиц
                try:
                    count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                    total_rows += count or 0
                except:
                    continue

            logger.info(f"Подключение проверено: {len(tables)} таблиц, ~{total_rows} записей")

        # Создаем запись в БД
        db_conn = crud.create_db_connection(db, user_id=current_user.id, conn_details=conn_details)
        logger.info(f"Пользователь {current_user.id} добавил SmartGPT-совместимое подключение {db_conn.id}")

        return db_conn

    except SQLAlchemyError as e:
        logger.error(f"Ошибка подключения к БД для SmartGPT: {e}")
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


# =============== ОСНОВНОЙ SMARTGPT DATAFRAME АНАЛИЗ ===============

@router.post("/reports/generate/{connection_id}", response_model=schemas.ReportInfo)
def generate_smart_analytics_report(
        connection_id: int,
        max_questions: Optional[int] = 15,
        analysis_type: Optional[str] = "standard",
        enable_gpt: Optional[bool] = True,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """
    Генерирует умный отчет с SmartGPT анализом

    Args:
        connection_id: ID подключения к БД
        max_questions: Максимальное количество вопросов для анализа (8-25)
        analysis_type: Тип анализа - quick, standard, comprehensive
        enable_gpt: Включить GPT анализ (по умолчанию True)
    """
    # Проверяем существование подключения
    db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
    if not db_conn:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    # Валидация параметров
    if analysis_type not in ["quick", "standard", "comprehensive"]:
        analysis_type = "standard"

    # Ограничиваем количество вопросов в зависимости от типа
    question_limits = {"quick": 8, "standard": 15, "comprehensive": 25}
    max_allowed = question_limits.get(analysis_type, 15)
    max_questions = min(max_questions or max_allowed, max_allowed)

    # Создаем запись отчета
    report_record = crud.create_report(
        db=db,
        user_id=current_user.id,
        connection_id=connection_id
    )

    # Выбираем и запускаем соответствующую задачу
    if analysis_type == "quick":
        logger.info(f"🚀 Запуск быстрого SmartGPT анализа для пользователя {current_user.id}")
        task = quick_dataframe_analysis.delay(connection_id, current_user.id, report_record.id)
    elif analysis_type == "comprehensive":
        logger.info(f"🧠 Запуск комплексного SmartGPT анализа для пользователя {current_user.id}")
        task = comprehensive_dataframe_analysis.delay(connection_id, current_user.id, report_record.id)
    else:  # standard
        logger.info(f"⚡ Запуск стандартного SmartGPT анализа для пользователя {current_user.id}")
        task = generate_dataframe_report.delay(connection_id, current_user.id, report_record.id, max_questions)

    # Обновляем запись с task_id
    report_record.task_id = task.id
    db.commit()
    db.refresh(report_record)

    logger.info(f"SmartGPT отчет {report_record.id} создан с task_id: {task.id}")
    return report_record


@router.post("/reports/smart-analysis/{connection_id}", response_model=schemas.ReportInfo)
def generate_smart_analysis_v2(
        connection_id: int,
        custom_questions: Optional[List[str]] = None,
        business_context: Optional[str] = None,
        priority_focus: Optional[str] = "business_insights",
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Расширенный SmartGPT анализ с кастомными вопросами и бизнес-контекстом"""

    db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
    if not db_conn:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    report_record = crud.create_report(
        db=db,
        user_id=current_user.id,
        connection_id=connection_id
    )

    # Определяем количество вопросов
    max_questions = len(custom_questions) if custom_questions else 15
    max_questions = min(max_questions, 30)  # Ограничение

    logger.info(f"🎯 Запуск кастомного SmartGPT анализа: {max_questions} вопросов, фокус: {priority_focus}")

    # Запускаем с дополнительными параметрами (через стандартную задачу)
    task = generate_dataframe_report.delay(connection_id, current_user.id, report_record.id, max_questions)

    report_record.task_id = task.id
    db.commit()
    db.refresh(report_record)

    return report_record


# =============== СТАТУС И ПОЛУЧЕНИЕ ОТЧЕТОВ ===============

@router.get("/reports/task/{task_id}/status")
def get_analysis_task_status(task_id: str):
    """Получает расширенный статус задачи SmartGPT анализа."""
    task_result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": task_result.status,
        "info": task_result.info
    }

    # Добавляем детальную информацию о прогрессе
    if task_result.info and isinstance(task_result.info, dict):
        response.update({
            "progress": task_result.info.get("progress", "Обработка..."),
            "stage": task_result.info.get("stage", "unknown"),
            "progress_percentage": task_result.info.get("progress_percentage", 0),
            "method": task_result.info.get("method", "dataframe_with_gpt"),
            "analysis_type": task_result.info.get("analysis_type", "standard"),
            "smartgpt_enabled": True
        })

        # Информация об ошибках
        if task_result.info.get("error"):
            response["error"] = task_result.info["error"]

        # Статистика успешного завершения
        if task_result.status == "SUCCESS":
            response.update({
                "report_id": task_result.info.get("report_id"),
                "questions_processed": task_result.info.get("questions_processed", 0),
                "successful_analyses": task_result.info.get("successful_analyses", 0),
                "gpt_analyses": task_result.info.get("gpt_analyses", 0),
                "tables_loaded": task_result.info.get("tables_loaded", 0)
            })

    logger.info(f"Запрошен статус SmartGPT задачи {task_id}: {task_result.status}")
    return response


@router.get("/reports/list/", response_model=List[schemas.ReportInfo])
def get_user_reports(
        limit: Optional[int] = 10,
        offset: Optional[int] = 0,
        status_filter: Optional[str] = None,
        method_filter: Optional[str] = None,
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

    # Фильтруем по методу если указан
    if method_filter:
        filtered_reports = []
        for report in reports:
            if report.results and isinstance(report.results, dict):
                method = report.results.get("method", "")
                if method_filter.lower() in method.lower():
                    filtered_reports.append(report)
        reports = filtered_reports

    logger.info(f"Пользователь {current_user.id} запросил список отчетов: {len(reports)} найдено")
    return reports


@router.get("/reports/{report_id}", response_model=schemas.Report)
def get_smart_report_by_id(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает готовый SmartGPT отчет по ID."""
    report = crud.get_report_by_id(db, report_id=report_id, user_id=current_user.id)

    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")

    logger.info(f"Пользователь {current_user.id} получил SmartGPT отчет {report_id}")

    # Обеспечиваем правильный формат results
    if isinstance(report.results, str):
        try:
            report.results = json.loads(report.results)
        except json.JSONDecodeError:
            report.results = {"error": "Некорректные данные отчета в базе данных."}

    return report


# =============== СПЕЦИАЛИЗИРОВАННЫЕ АНАЛИЗЫ ===============

@router.post("/quick-analysis/{connection_id}", response_model=schemas.ReportInfo)
def run_quick_smart_analysis(
        connection_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Быстрый SmartGPT анализ (8 ключевых вопросов, ~10 минут)"""

    db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
    if not db_conn:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    report_record = crud.create_report(
        db=db,
        user_id=current_user.id,
        connection_id=connection_id
    )

    logger.info(f"⚡ Быстрый SmartGPT анализ для пользователя {current_user.id}")
    task = quick_dataframe_analysis.delay(connection_id, current_user.id, report_record.id)

    report_record.task_id = task.id
    db.commit()
    db.refresh(report_record)

    return report_record


@router.post("/comprehensive-analysis/{connection_id}", response_model=schemas.ReportInfo)
def run_comprehensive_smart_analysis(
        connection_id: int,
        include_predictive: Optional[bool] = True,
        deep_correlation_analysis: Optional[bool] = True,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Глубокий SmartGPT анализ (25 вопросов, предиктивная аналитика, ~45 минут)"""

    db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
    if not db_conn:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    report_record = crud.create_report(
        db=db,
        user_id=current_user.id,
        connection_id=connection_id
    )

    logger.info(f"🧠 Комплексный SmartGPT анализ для пользователя {current_user.id}")
    task = comprehensive_dataframe_analysis.delay(connection_id, current_user.id, report_record.id)

    report_record.task_id = task.id
    db.commit()
    db.refresh(report_record)

    return report_record


# =============== ДИАГНОСТИКА И МОНИТОРИНГ ===============

@router.get("/connection-health/{connection_id}")
def check_connection_health(
        connection_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Проверяет готовность базы данных к SmartGPT анализу."""

    db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
    if not db_conn:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    try:
        connection_string = crud.get_decrypted_connection_string(db, connection_id, current_user.id)
        engine = create_engine(connection_string, connect_args={'connect_timeout': 10})

        with engine.connect() as conn:
            from sqlalchemy import inspect, text
            inspector = inspect(engine)

            tables = [t for t in inspector.get_table_names() if t not in ['alembic_version', 'django_migrations']]

            # Анализируем готовность к SmartGPT
            smartgpt_readiness = 0
            total_rows = 0
            business_ready_tables = 0

            for table in tables[:10]:  # Анализируем до 10 таблиц
                try:
                    count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
                    total_rows += count

                    if count > 0:
                        # Проверяем бизнес-готовность по названиям колонок
                        columns = [col['name'] for col in inspector.get_columns(table)]
                        column_names = " ".join(columns).lower()

                        if any(word in column_names for word in
                               ['price', 'amount', 'customer', 'user', 'sale', 'order', 'date', 'id']):
                            business_ready_tables += 1
                except:
                    continue

            # Оценка SmartGPT готовности
            if len(tables) > 0:
                smartgpt_readiness += 30
            if business_ready_tables > 0:
                smartgpt_readiness += 40
            if total_rows > 1000:
                smartgpt_readiness += 30

            health_status = {
                "connection": True,
                "smartgpt_ready": smartgpt_readiness >= 70,
                "readiness_score": smartgpt_readiness,
                "total_tables": len(tables),
                "business_ready_tables": business_ready_tables,
                "total_rows": total_rows,
                "estimated_analysis_time": (
                    "Быстро (5-15 мин)" if total_rows < 10000 else
                    "Средне (15-30 мин)" if total_rows < 100000 else
                    "Долго (30-60 мин)"
                ),
                "recommended_analysis": (
                    "comprehensive" if smartgpt_readiness >= 90 else
                    "standard" if smartgpt_readiness >= 70 else
                    "quick"
                ),
                "smartgpt_features": [
                    "business_insights" if business_ready_tables > 0 else None,
                    "correlation_analysis" if business_ready_tables > 1 else None,
                    "trend_analysis" if total_rows > 100 else None
                ]
            }

            # Убираем None значения
            health_status["smartgpt_features"] = [f for f in health_status["smartgpt_features"] if f]

            return health_status

    except Exception as e:
        logger.error(f"Ошибка проверки здоровья подключения {connection_id}: {e}")
        return {
            "connection": False,
            "smartgpt_ready": False,
            "readiness_score": 0,
            "error": str(e)
        }


@router.get("/system-stats")
def get_smartgpt_system_stats(
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает статистику системы SmartGPT."""

    # Проверяем активные задачи Celery
    try:
        i = celery_app.control.inspect()
        active_tasks = i.active()
        scheduled_tasks = i.scheduled()

        active_count = sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0
        scheduled_count = sum(len(tasks) for tasks in scheduled_tasks.values()) if scheduled_tasks else 0

        # Подсчитываем SmartGPT задачи
        smartgpt_active = 0
        if active_tasks:
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    if any(keyword in task.get('name', '').lower()
                           for keyword in ['dataframe', 'smart', 'gpt']):
                        smartgpt_active += 1

    except Exception as e:
        logger.warning(f"Не удалось получить статистику Celery: {e}")
        active_count = 0
        scheduled_count = 0
        smartgpt_active = 0

    return {
        "smartgpt_version": "2.0",
        "smartgpt_enabled": True,
        "active_tasks": active_count,
        "active_smartgpt_tasks": smartgpt_active,
        "scheduled_tasks": scheduled_count,
        "user_id": current_user.id,

        "analysis_capabilities": {
            "dataframe_analysis": True,
            "gpt_business_insights": True,
            "correlation_analysis": True,
            "anomaly_detection": True,
            "trend_analysis": True,
            "predictive_analytics": True
        },

        "supported_analysis_types": ["quick", "standard", "comprehensive"],
        "max_questions_per_analysis": 25,
        "supported_databases": ["PostgreSQL", "MySQL", "SQLite", "SQL Server"],

        "performance_info": {
            "avg_analysis_time_minutes": {
                "quick": "5-10",
                "standard": "15-20",
                "comprehensive": "30-45"
            },
            "max_dataframe_size_mb": 2048,
            "max_tables_per_analysis": 20
        }
    }


# =============== ОБРАТНАЯ СВЯЗЬ ===============

@router.post("/reports/feedback/{report_id}")
def submit_report_feedback(
        report_id: int,
        rating: int,
        comment: Optional[str] = None,
        useful_sections: Optional[List[str]] = None,
        smartgpt_quality: Optional[int] = None,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Отправляет обратную связь по SmartGPT отчету."""

    if rating < 1 or rating > 5:
        raise HTTPException(status_code=400, detail="Рейтинг должен быть от 1 до 5")

    if smartgpt_quality and (smartgpt_quality < 1 or smartgpt_quality > 5):
        raise HTTPException(status_code=400, detail="Оценка SmartGPT должна быть от 1 до 5")

    try:
        feedback_data = {
            "rating": rating,
            "comment": comment,
            "useful_sections": useful_sections or [],
            "smartgpt_quality": smartgpt_quality,
            "smartgpt_feedback": True
        }

        feedback_record = crud.create_report_feedback(
            db,
            report_id=report_id,
            user_id=current_user.id,
            feedback_data=feedback_data
        )

        logger.info(f"SmartGPT обратная связь получена для отчета {report_id} от пользователя {current_user.id}")
        return {"status": "success", "feedback_id": feedback_record.id}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# =============== LEGACY ПОДДЕРЖКА ===============

@router.post("/reports/generate-advanced/{connection_id}", response_model=schemas.ReportInfo)
def generate_legacy_advanced_report(
        connection_id: int,
        max_questions: Optional[int] = 12,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Legacy эндпоинт - перенаправляет на SmartGPT анализ."""

    logger.warning(f"Использование legacy эндпоинта, перенаправляем на SmartGPT для пользователя {current_user.id}")

    # Перенаправляем на стандартный SmartGPT анализ
    return generate_smart_analytics_report(
        connection_id=connection_id,
        max_questions=max_questions,
        analysis_type="standard",
        enable_gpt=True,
        db=db,
        current_user=current_user
    )


logger.info("SmartGPT DataFrame Analytics Router полностью загружен")
