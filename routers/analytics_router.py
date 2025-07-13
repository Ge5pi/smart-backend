# routers/analytics_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

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


# ЗАГЛУШКА для следующего этапа
@router.post("/reports/generate/{connection_id}", response_model=schemas.ReportInfo)
def generate_report_placeholder(
        connection_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Запускает процесс генерации отчета (пока заглушка)."""
    # Проверяем, что подключение существует и принадлежит пользователю
    db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
    if not db_conn:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    # TODO

    # На Фазе 2 здесь будет вызов задачи Celery
    # task = generate_full_report.delay(connection_id, current_user.id)
    # А пока создаем фейковый отчет

    task_id_placeholder = f"fake_task_{connection_id}"
    report = crud.create_report(db, user_id=current_user.id, connection_id=connection_id, task_id=task_id_placeholder)

    return report
