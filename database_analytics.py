import io

from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session
import logging

# ИЗМЕНЕНИЕ: Импортируем задачу Celery
from starlette.responses import StreamingResponse

from celery_worker import run_db_analysis_task
import auth
import crud
import database
import models
import schemas
from pdf_generator import generate_pdf_report

database_router = APIRouter(prefix="/analytics/database")


@database_router.post("/analyze")
async def analyze_database(
        connectionString: str = Form(...),
        dbType: str = Form(...),
        alias: str = Form(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    if dbType not in ['postgres', 'sqlserver']:
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип базы данных.")

    connection = crud.create_database_connection(db, user_id=current_user.id, connection_string=connectionString,
                                                 db_type=dbType, alias=alias)

    report = crud.create_report(db, user_id=current_user.id, connection_id=connection.id, status="queued")
    logging.warning(f"Отчет ID:{report.id} добавлен в очередь.")

    run_db_analysis_task.delay(
        report_id=report.id,
        user_id=current_user.id,
        connection_string=connectionString,
        db_type=dbType
    )

    return {"report_id": report.id, "message": "Анализ запущен в фоновом режиме. Отчет будет готов в ближайшее время."}


@database_router.get("/connections", response_model=list[schemas.DatabaseConnection])
async def get_user_connections(
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    return crud.get_database_connections_by_user_id(db, user_id=current_user.id)


@database_router.get("/reports/{report_id}")
async def get_report_details(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    report = crud.get_report_by_id(db, report_id=report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден.")

    if report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Недостаточно прав для просмотра этого отчета.")

    return report


@database_router.get("/reports", response_model=list[schemas.Report])
async def get_user_reports(
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_active_user)
):
    return crud.get_reports_by_user_id(db, user_id=current_user.id)


@database_router.get("/reports/{report_id}/pdf")
async def download_report_pdf(
        report_id: int,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    report = crud.get_report_by_id(db, report_id=report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Отчет не найден.")

    if report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Недостаточно прав для просмотра этого отчета.")

    try:
        pdf_buffer = generate_pdf_report(report)

        return StreamingResponse(
            io.BytesIO(pdf_buffer.read()),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=report_{report_id}.pdf"
            }
        )
    except Exception as e:
        logging.error(f"Ошибка при генерации PDF для отчета {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при генерации PDF отчета.")