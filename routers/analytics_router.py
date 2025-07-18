from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
import logging
import json
from datetime import datetime

import database
import models
import schemas
import crud
import auth
from tasks import generate_enhanced_report

# Настройка логирования
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analytics",
    tags=["analytics"]
)


@router.get("/connections/", response_model=list[schemas.DatabaseConnectionInfo])
def get_user_connections(
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает все подключения пользователя."""
    connections = crud.get_db_connections_by_user(db, user_id=current_user.id)
    logger.info(f"Пользователь {current_user.id} запросил список подключений: {len(connections)} найдено")
    return connections


@router.post("/reports/create")
def create_analysis_report(
        connection_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Создает новый отчет анализа для указанного подключения."""
    # Проверяем, что подключение существует и принадлежит пользователю
    connection = crud.get_db_connection_by_id(db, connection_id, current_user.id)
    if not connection:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    # Создаем новый отчет
    report = crud.create_report(db, current_user.id, connection_id)

    # Запускаем асинхронную задачу анализа
    task = generate_enhanced_report.delay(
        report_id=report.id,
        connection_id=connection_id,
        user_id=current_user.id
    )

    # Обновляем отчет с task_id
    crud.update_report(db, report.id, "PROCESSING", {})
    report.task_id = task.id
    db.commit()

    logger.info(f"Создан отчет {report.id} для пользователя {current_user.id} с задачей {task.id}")

    return {
        "report_id": report.id,
        "task_id": task.id,
        "status": "PROCESSING",
        "message": "Анализ запущен. Проверьте статус через несколько минут."
    }


@router.get("/reports/{report_id}")
def get_report_by_id(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает отчет по ID с полной информацией."""
    report = crud.get_report_by_id(db, report_id=report_id, user_id=current_user.id)
    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")

    logger.info(f"Пользователь {current_user.id} получил данные отчета {report_id} со статусом {report.status}")

    # Получаем данные из поля content
    results_data = report.content

    # Защита на случай, если из БД данные пришли в виде строки
    if isinstance(results_data, str):
        try:
            results_data = json.loads(results_data)
        except json.JSONDecodeError:
            logger.error(f"Malformed JSON in report {report_id}: {results_data}")
            results_data = {"error": "Malformed report data in database."}

    # Если данные отсутствуют, возвращаем базовую структуру
    if not results_data:
        results_data = {
            "executive_summary": "Анализ еще не завершен",
            "detailed_findings": [],
            "recommendations": [],
            "domain_context": {
                "domain_type": "unknown",
                "confidence": 0.0,
                "key_entities": [],
                "business_metrics": []
            },
            "ml_insights": {
                "total_patterns": 0,
                "pattern_types": {},
                "high_confidence_patterns": []
            },
            "analysis_stats": {
                "questions_processed": 0,
                "successful_findings": 0,
                "ml_patterns_found": 0,
                "tables_coverage": 0,
                "tables_analyzed": 0
            },
            "diversity_report": {
                "total_tables": 0,
                "analyzed_tables": 0,
                "coverage_percentage": 0,
                "underanalyzed_tables": []
            }
        }

    return {
        "id": report.id,
        "status": report.status,
        "created_at": report.created_at.isoformat(),
        "task_id": report.task_id,
        "results": results_data  # Маппим content в results для фронтенда
    }


@router.get("/reports/status/{task_id}")
def get_task_status(task_id: str):
    """Получает статус выполнения задачи анализа."""
    from celery.result import AsyncResult

    try:
        # Получаем статус задачи из Celery
        task_result = AsyncResult(task_id)

        if task_result.state == "PENDING":
            return {
                "task_id": task_id,
                "status": "PENDING",
                "progress": "Задача в очереди",
                "stage": "waiting",
                "progress_percentage": 0,
                "current_question": "",
                "diversity_report": {
                    "total_tables": 0,
                    "analyzed_tables": 0,
                    "coverage_percentage": 0,
                    "underanalyzed_tables": []
                },
                "summary": {
                    "questions_processed": 0,
                    "findings_count": 0,
                    "ml_patterns_found": 0,
                    "domain_detected": "unknown"
                }
            }

        elif task_result.state == "PROGRESS":
            # Получаем информацию о прогрессе с защитой от некорректных данных
            info = task_result.info if isinstance(task_result.info, dict) else {}

            # Безопасное получение значений с fallback
            progress = info.get("progress", "Выполняется анализ") if isinstance(info, dict) else "Выполняется анализ"
            stage = info.get("stage", "processing") if isinstance(info, dict) else "processing"
            progress_percentage = info.get("progress_percentage", 0) if isinstance(info, dict) else 0
            current_question = info.get("current_question", "") if isinstance(info, dict) else ""

            # Безопасное получение diversity_report
            diversity_report = info.get("diversity_report", {}) if isinstance(info, dict) else {}
            if not isinstance(diversity_report, dict):
                diversity_report = {}

            diversity_report_safe = {
                "total_tables": diversity_report.get("total_tables", 0),
                "analyzed_tables": diversity_report.get("analyzed_tables", 0),
                "coverage_percentage": diversity_report.get("coverage_percentage", 0),
                "underanalyzed_tables": diversity_report.get("underanalyzed_tables", [])
            }

            # Безопасное получение summary
            summary = info.get("summary", {}) if isinstance(info, dict) else {}
            if not isinstance(summary, dict):
                summary = {}

            summary_safe = {
                "questions_processed": summary.get("questions_processed", 0),
                "findings_count": summary.get("findings_count", 0),
                "ml_patterns_found": summary.get("ml_patterns_found", 0),
                "domain_detected": summary.get("domain_detected", "unknown")
            }

            return {
                "task_id": task_id,
                "status": "PROGRESS",
                "progress": progress,
                "stage": stage,
                "progress_percentage": progress_percentage,
                "current_question": current_question,
                "diversity_report": diversity_report_safe,
                "summary": summary_safe
            }

        elif task_result.state == "SUCCESS":
            return {
                "task_id": task_id,
                "status": "SUCCESS",
                "progress": "Анализ завершен",
                "stage": "completed",
                "progress_percentage": 100,
                "current_question": "",
                "diversity_report": {
                    "total_tables": 0,
                    "analyzed_tables": 0,
                    "coverage_percentage": 100,
                    "underanalyzed_tables": []
                },
                "summary": {
                    "questions_processed": 0,
                    "findings_count": 0,
                    "ml_patterns_found": 0,
                    "domain_detected": "completed"
                }
            }

        elif task_result.state == "FAILURE":
            # Безопасное получение информации об ошибке
            error_info = task_result.info if task_result.info is not None else "Неизвестная ошибка"
            error_message = str(error_info) if error_info else "Неизвестная ошибка"

            return {
                "task_id": task_id,
                "status": "FAILURE",
                "progress": "Ошибка выполнения",
                "stage": "failed",
                "progress_percentage": 0,
                "current_question": "",
                "error": error_message,
                "diversity_report": {
                    "total_tables": 0,
                    "analyzed_tables": 0,
                    "coverage_percentage": 0,
                    "underanalyzed_tables": []
                },
                "summary": {
                    "questions_processed": 0,
                    "findings_count": 0,
                    "ml_patterns_found": 0,
                    "domain_detected": "failed"
                }
            }

        else:
            return {
                "task_id": task_id,
                "status": task_result.state,
                "progress": "Неизвестный статус",
                "stage": "unknown",
                "progress_percentage": 0,
                "current_question": "",
                "diversity_report": {
                    "total_tables": 0,
                    "analyzed_tables": 0,
                    "coverage_percentage": 0,
                    "underanalyzed_tables": []
                },
                "summary": {
                    "questions_processed": 0,
                    "findings_count": 0,
                    "ml_patterns_found": 0,
                    "domain_detected": "unknown"
                }
            }

    except Exception as e:
        logger.error(f"Ошибка получения статуса задачи {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")


@router.get("/reports/list/", response_model=List[schemas.ReportInfo])
def get_user_reports_list(
        limit: Optional[int] = Query(10, ge=1, le=100),
        offset: Optional[int] = Query(0, ge=0),
        status_filter: Optional[str] = Query(None),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает список отчетов пользователя с фильтрацией."""
    try:
        reports = crud.get_user_reports(
            db=db,
            user_id=current_user.id,
            limit=limit,
            offset=offset,
            status_filter=status_filter
        )

        logger.info(f"Пользователь {current_user.id} запросил список отчетов: {len(reports)} найдено")
        return reports
    except Exception as e:
        logger.error(f"Ошибка получения списка отчетов: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения списка отчетов")


@router.post("/reports/feedback/{report_id}")
def submit_report_feedback(
        report_id: int,
        feedback_data: schemas.FeedbackCreate,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Отправляет обратную связь по отчету для улучшения будущих анализов."""
    try:
        feedback = crud.create_report_feedback(
            db=db,
            report_id=report_id,
            user_id=current_user.id,
            feedback_data=feedback_data.dict()
        )

        logger.info(f"Получена обратная связь для отчета {report_id} от пользователя {current_user.id}")

        return {
            "message": "Спасибо за обратную связь! Она поможет улучшить будущие анализы.",
            "feedback_id": feedback.id
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка создания обратной связи: {e}")
        raise HTTPException(status_code=500, detail="Ошибка сохранения обратной связи")


@router.get("/reports/{report_id}/feedback")
def get_report_feedback(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает обратную связь по отчету."""
    # Проверяем, что отчет существует и принадлежит пользователю
    report = crud.get_report_by_id(db, report_id, current_user.id)
    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")

    try:
        feedbacks = crud.get_report_feedbacks(db, report_id)
        return {
            "report_id": report_id,
            "feedbacks": feedbacks
        }
    except Exception as e:
        logger.error(f"Ошибка получения обратной связи: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения обратной связи")


@router.delete("/reports/{report_id}")
def delete_report(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Удаляет отчет пользователя."""
    report = crud.get_report_by_id(db, report_id, current_user.id)
    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")

    try:
        # Удаляем отчет
        db.delete(report)
        db.commit()

        logger.info(f"Удален отчет {report_id} пользователя {current_user.id}")

        return {"message": "Отчет успешно удален"}
    except Exception as e:
        logger.error(f"Ошибка удаления отчета {report_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Ошибка удаления отчета")


@router.get("/health")
def health_check():
    """Проверка работоспособности сервиса аналитики."""
    return {
        "status": "healthy",
        "service": "analytics",
        "timestamp": datetime.now().isoformat()
    }
