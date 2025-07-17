# analytics_router.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from celery.result import AsyncResult

from tasks import generate_advanced_report
from celery_worker import celery_app
import crud
import schemas
import models
import auth
import database

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics Engine"],
    dependencies=[Depends(auth.get_current_active_user)]
)


@router.post("/connections/", response_model=schemas.DatabaseConnectionInfo)
def add_database_connection(conn_details: schemas.DatabaseConnectionCreate, db: Session = Depends(database.get_db),
                            current_user: models.User = Depends(auth.get_current_active_user)):
    try:
        engine = create_engine(conn_details.connection_string, connect_args={'connect_timeout': 10})
        connection = engine.connect()
        connection.close()
    except SQLAlchemyError as e:
        raise HTTPException(status_code=400, detail=f"Неверная строка подключения: {e}")
    db_conn = crud.create_db_connection(db, user_id=current_user.id, conn_details=conn_details)
    return db_conn


@router.get("/connections/", response_model=list[schemas.DatabaseConnectionInfo])
def get_user_connections(db: Session = Depends(database.get_db),
                         current_user: models.User = Depends(auth.get_current_active_user)):
    return crud.get_db_connections_by_user(db, user_id=current_user.id)


@router.post("/reports/generate/{connection_id}", response_model=schemas.ReportInfo)
def generate_report(connection_id: int, db: Session = Depends(database.get_db),
                    current_user: models.User = Depends(auth.get_current_active_user)):
    db_conn = crud.get_db_connection_by_id(db, connection_id, user_id=current_user.id)
    if not db_conn:
        raise HTTPException(status_code=404, detail="Подключение не найдено")

    report_record = crud.create_report(db=db, user_id=current_user.id, connection_id=connection_id)
    task = generate_advanced_report.delay(connection_id, current_user.id, report_record.id)
    report_record.task_id = task.id
    db.commit()
    db.refresh(report_record)
    return report_record


@router.get("/reports/status/{task_id}")
def get_report_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    response = {"task_id": task_id, "status": task_result.status, "info": task_result.info}
    return response


@router.get("/reports/{report_id}", response_model=schemas.Report)
def get_report_by_id(report_id: int, db: Session = Depends(database.get_db),
                     current_user: models.User = Depends(auth.get_current_active_user)):
    report = crud.get_report_by_id(db, report_id=report_id, user_id=current_user.id)
    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден")
    if report.status != 'COMPLETED':
        raise HTTPException(status_code=202, detail=f"Отчет еще в процессе генерации. Статус: {report.status}")
    return report
