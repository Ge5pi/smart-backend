# routers/analytics_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from tasks import generate_full_report


import crud, schemas, models, auth, database

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics Engine"],
    dependencies=[Depends(auth.get_current_active_user)]
)


@router.post("/connections/", response_model=schemas.DatabaseConnectionInfo)
def add_database_connection(
        conn_details: schemas.DatabaseConnectionCreate,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Добавляет новое подключение к БД, проверяя его валидность."""
    try:
        # Попытка создать движок для проверки строки подключения
        engine = create_engine(conn_details.connection_string)
        connection = engine.connect()
        connection.close()
    except SQLAlchemyError as e:
        raise HTTPException(status_code=400, detail=f"Неверная строка подключения: {e}")

    db_conn = crud.create_db_connection(db, user_id=current_user.id, conn_details=conn_details)
    return db_conn


@router.get("/connections/", response_model=list[schemas.DatabaseConnectionInfo])
def get_user_connections(
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Возвращает список всех подключений пользователя."""
    return crud.get_db_connections_by_user(db, user_id=current_user.id)


@router.post("/reports/generate/{connection_id}", response_model=schemas.ReportInfo)
def generate_report(
    connection_id: int,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    """Запускает асинхронную задачу генерации отчета."""
    db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
    if not db_conn:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    # Создаем запись в БД для отчета со статусом PENDING
    # Это нужно, чтобы получить report_id для передачи в задачу
    initial_report = crud.create_report(
        db, user_id=current_user.id, connection_id=connection_id
    )

    # Запускаем нашу фоновую задачу с помощью .delay()
    # .delay() - это сокращенный способ вызова .apply_async()
    task = generate_full_report.delay(connection_id, current_user.id, initial_report.id)

    # Обновляем запись в БД, добавляя реальный ID задачи от Celery
    initial_report.task_id = task.id
    db.commit()
    db.refresh(initial_report)

    return initial_report


# routers/analytics_router.py
from celery.result import AsyncResult
from tasks import celery_app  # Импортируем наш celery_app


# ...

@router.get("/reports/status/{task_id}")
def get_report_status(task_id: str):
    """Возвращает статус выполнения задачи Celery."""
    task_result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": task_result.status,  # Статус выполнения (PENDING, SUCCESS, FAILURE)
        "info": task_result.info,  # Дополнительная информация (то, что мы передаем в meta)
    }
    return response


@router.get("/reports/{report_id}", response_model=schemas.Report)
def get_report_by_id(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Возвращает готовый отчет из БД по его ID."""
    report = crud.get_report_by_id(db, report_id=report_id, user_id=current_user.id)
    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")
    return report