import io

from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session
import logging
from starlette.responses import StreamingResponse, Response

from celery_worker import run_db_analysis_task
import auth
import crud
import database
import models
import schemas
from pdf_generator import generate_pdf_report
from config import REPORT_LIMIT

database_router = APIRouter(prefix="/analytics/database")


@database_router.post("/analyze")
async def analyze_database(
        connectionString: str = Form(...),
        dbType: str = Form(...),
        alias: str = Form(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user),
        language: str = Form("en")
):
    if (not current_user.is_active) and current_user.reports_used >= REPORT_LIMIT:
        raise HTTPException(
            status_code=403,
            detail="Вы использовали все бесплатные генерации отчетов. Пожалуйста, перейдите на платный тариф."
        )

    if dbType not in ['postgres', 'sqlserver']:
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип базы данных.")

    connection = crud.create_database_connection(db, user_id=current_user.id, connection_string=connectionString,
                                                 db_type=dbType, alias=alias)

    report = crud.create_report(db, user_id=current_user.id, connection_id=connection.id, status="queued")
    logging.warning(f"Отчет ID:{report.id} добавлен в очередь.")

    crud.increment_usage_counter(db, user=current_user, counter_type='reports')
    run_db_analysis_task.delay(
        report.id,
        current_user.id,
        connectionString,
        dbType,
        language
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

    if report.status != 'completed':
        raise HTTPException(status_code=400, detail="Отчет еще не готов для скачивания.")

    try:
        pdf_buffer = generate_pdf_report(report)

        pdf_content = pdf_buffer.getvalue()
        if len(pdf_content) == 0:
            raise HTTPException(status_code=500, detail="Сгенерированный PDF файл пустой.")

        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=\"report_{report_id}.pdf\"",
                "Content-Length": str(len(pdf_content)),
                "Cache-Control": "no-cache"
            }
        )
    except Exception as e:
        logging.error(f"Ошибка при генерации PDF для отчета {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации PDF отчета: {str(e)}")