# analytics_router.py
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from celery.result import AsyncResult
from tasks import (
    generate_enhanced_report,
    quick_ml_analysis,
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
    tags=["Enhanced Analytics Engine"],
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


# =============== ГЕНЕРАЦИЯ ОТЧЕТОВ ===============

@router.post("/reports/generate/{connection_id}", response_model=schemas.ReportInfo)
def generate_enhanced_analytics_report(
        connection_id: int,
        max_questions: Optional[int] = 15,
        enable_feedback: Optional[bool] = True,
        analysis_type: Optional[str] = "enhanced",  # enhanced, quick, legacy
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """
    Генерирует улучшенный отчет с ML-анализом и адаптивной обратной связью.

    Args:
        connection_id: ID подключения к БД
        max_questions: Максимальное количество вопросов для анализа (по умолчанию 15)
        enable_feedback: Включить адаптивную обратную связь (по умолчанию True)
        analysis_type: Тип анализа - enhanced, quick, legacy
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
        logger.info(f"Запуск быстрого ML-анализа для пользователя {current_user.id}")
        task = quick_ml_analysis.delay(connection_id, current_user.id, report_record.id)

    elif analysis_type == "legacy":
        logger.info(f"Запуск legacy анализа для пользователя {current_user.id}")
        task = generate_advanced_report.delay(connection_id, current_user.id, report_record.id)

    else:  # enhanced (по умолчанию)
        logger.info(f"Запуск улучшенного анализа для пользователя {current_user.id}")
        task = generate_enhanced_report.delay(
            connection_id,
            current_user.id,
            report_record.id,
            max_questions,
            enable_feedback
        )

    # Обновляем запись с task_id
    report_record.task_id = task.id
    db.commit()
    db.refresh(report_record)

    logger.info(f"Отчет {report_record.id} создан с task_id: {task.id}")
    return report_record


# Сохраняем старый эндпоинт для обратной совместимости
@router.post("/reports/generate-legacy/{connection_id}", response_model=schemas.ReportInfo)
def generate_legacy_report(
        connection_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Legacy эндпоинт для старых клиентов."""
    return generate_enhanced_analytics_report(
        connection_id=connection_id,
        max_questions=12,
        enable_feedback=False,
        analysis_type="legacy",
        db=db,
        current_user=current_user
    )


# =============== СТАТУС И ПОЛУЧЕНИЕ ОТЧЕТОВ ===============

@router.get("/reports/status/{task_id}")
def get_enhanced_report_status(task_id: str):
    """
    Получает расширенный статус задачи анализа.
    """
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
            "current_question": task_result.info.get("question", ""),
            "diversity_report": task_result.info.get("diversity_report", {}),
            "summary": task_result.info.get("summary", {})
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
def get_enhanced_report_by_id(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает готовый отчет по ID с полной информацией."""
    report = crud.get_report_by_id(db, report_id=report_id, user_id=current_user.id)
    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")

    logger.info(f"Пользователь {current_user.id} получил данные отчета {report_id}")

    # Ensure results are properly loaded as a dict before returning
    if isinstance(report.results, str):
        try:
            report.results = json.loads(report.results)
        except json.JSONDecodeError:
            report.results = {"error": "Malformed report data in database."}

    return report


@router.post("/reports/feedback/{report_id}", response_model=schemas.FeedbackResponse)
def submit_report_feedback(
        report_id: int,
        feedback_data: schemas.FeedbackCreate,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Отправляет обратную связь по отчету для улучшения будущих анализов."""
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
        from services.report_agents import get_database_health_check
        health_check = get_database_health_check(engine)
        return health_check
    except Exception as e:
        logger.error(f"Ошибка проверки здоровья БД {connection_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка проверки подключения: {e}")


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
        "available_analysis_types": ["enhanced", "quick", "legacy"],
        "max_questions_limit": 50,
        "supported_features": [
            "ml_pattern_detection",
            "domain_analysis",
            "adaptive_feedback",
            "advanced_validation",
            "intelligent_prioritization"
        ]
    }
    return stats