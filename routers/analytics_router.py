# analytics_router.py

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
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
from typing import Optional
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


@router.get("/connections/", response_model=list[schemas.DatabaseConnectionInfo])
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

    Возвращает детальную информацию о прогрессе, включая:
    - Текущий этап анализа
    - Процент выполнения
    - ML-инсайты в реальном времени
    - Информацию о разнообразии анализа
    """

    task_result = AsyncResult(task_id, app=celery_app)

    # Базовая информация о задаче
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "info": task_result.info
    }

    # Расширенная информация для улучшенных задач
    if task_result.info and isinstance(task_result.info, dict):
        # Добавляем детали прогресса
        response.update({
            "progress": task_result.info.get("progress", "Неизвестно"),
            "stage": task_result.info.get("stage", "unknown"),
            "progress_percentage": task_result.info.get("progress_percentage", 0),
            "current_question": task_result.info.get("question", ""),
            "diversity_report": task_result.info.get("diversity_report", {}),
            "summary": task_result.info.get("summary", {})
        })

        # Добавляем информацию об ошибках
        if task_result.info.get("error"):
            response["error"] = task_result.info["error"]

    logger.info(f"Запрошен статус задачи {task_id}: {task_result.status}")
    return response


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

    if report.status == 'FAILED':
        logger.warning(f"Запрошен неуспешный отчет {report_id}")
        raise HTTPException(status_code=422, detail=f"Отчет завершился с ошибкой: {report.results}")

    if report.status != 'COMPLETED':
        raise HTTPException(
            status_code=202,
            detail=f"Отчет еще в процессе генерации. Статус: {report.status}"
        )

    logger.info(f"Пользователь {current_user.id} получил отчет {report_id}")
    return report


# =============== ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ===============

@router.get("/reports/list/", response_model=list[schemas.ReportInfo])
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


@router.delete("/reports/{report_id}")
def delete_report(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Удаляет отчет пользователя."""

    report = crud.get_report_by_id(db, report_id=report_id, user_id=current_user.id)

    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")

    # Останавливаем задачу если она еще выполняется
    if report.task_id and report.status in ['PENDING', 'PROGRESS']:
        celery_app.control.revoke(report.task_id, terminate=True)

    crud.delete_report(db, report_id=report_id)

    logger.info(f"Пользователь {current_user.id} удалил отчет {report_id}")
    return {"message": "Отчет успешно удален"}


@router.post("/reports/feedback/{report_id}")
def submit_report_feedback(
        report_id: int,
        feedback_data: dict,  # {rating: int, comment: str, useful_sections: list}
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Отправляет обратную связь по отчету для улучшения будущих анализов."""

    report = crud.get_report_by_id(db, report_id=report_id, user_id=current_user.id)

    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")

    # Сохраняем обратную связь
    crud.create_report_feedback(
        db,
        report_id=report_id,
        user_id=current_user.id,
        feedback_data=feedback_data
    )

    logger.info(f"Получена обратная связь для отчета {report_id} от пользователя {current_user.id}")
    return {"message": "Спасибо за обратную связь! Она поможет улучшить будущие анализы."}


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

        # Используем функцию проверки здоровья из report_agents
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

    # Получаем статистику задач Celery
    active_tasks = celery_app.control.inspect().active()
    scheduled_tasks = celery_app.control.inspect().scheduled()

    stats = {
        "active_tasks": len(active_tasks) if active_tasks else 0,
        "scheduled_tasks": len(scheduled_tasks) if scheduled_tasks else 0,
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
