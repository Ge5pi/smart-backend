from typing import List

from fastapi import APIRouter, UploadFile, File, Depends, Query, Form
from sqlalchemy.orm import Session
import database
import models
import schemas

from main import vk_upload_file, vk_read_user_files, vk_analyze_existing_file, vk_impute_missing, vk_detect_outliers, \
    vk_encode_categorical, vk_download_cleaned_file, vk_get_paginated_preview
from tg_auth import verify_telegram_init_data

telegram_mini_app_router = APIRouter(
    prefix="/tg-api",
    tags=["Telegram Mini App"]
)


@telegram_mini_app_router.post("/upload/")
async def tg_upload_file(file: UploadFile = File(...), db: Session = Depends(database.get_db),
                         current_user: models.User = Depends(verify_telegram_init_data)):
    return await vk_upload_file(file, db, current_user)


@telegram_mini_app_router.get("/files/me", response_model=List[schemas.File])
def tg_read_user_files(db: Session = Depends(database.get_db),
                       current_user: models.User = Depends(verify_telegram_init_data)):
    return vk_read_user_files(db, current_user)


@telegram_mini_app_router.post("/analyze-existing/")
async def tg_analyze_existing_file(file_id: str = Form(...), db: Session = Depends(database.get_db),
                                   current_user: models.User = Depends(verify_telegram_init_data)):
    return await vk_analyze_existing_file(file_id, db, current_user)


@telegram_mini_app_router.post("/impute-missing/")
async def tg_impute_missing(file_id: str = Form(...), columns: str = Form(...), db: Session = Depends(database.get_db),
                            current_user: models.User = Depends(verify_telegram_init_data)):
    return await vk_impute_missing(file_id, columns, db, current_user)


@telegram_mini_app_router.post("/outliers/")
async def tg_detect_outliers(file_id: str = Form(...), columns: str = Form(...), db: Session = Depends(database.get_db),
                             current_user: models.User = Depends(verify_telegram_init_data)):
    return await vk_detect_outliers(file_id, columns, db, current_user)


@telegram_mini_app_router.post("/encode-categorical/")
async def tg_encode_categorical(file_id: str = Form(...), columns: str = Form(...),
                                db: Session = Depends(database.get_db),
                                current_user: models.User = Depends(verify_telegram_init_data)):
    return await vk_encode_categorical(file_id, columns, db, current_user)


@telegram_mini_app_router.get("/download-cleaned/{file_id}")
async def tg_download_cleaned_file(file_id: str, db: Session = Depends(database.get_db),
                                   current_user: models.User = Depends(verify_telegram_init_data)):
    return await vk_download_cleaned_file(file_id, db, current_user)


@telegram_mini_app_router.get("/preview/{file_id}")
async def tg_get_paginated_preview(file_id: str, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200),
                                   db: Session = Depends(database.get_db),
                                   current_user: models.User = Depends(verify_telegram_init_data)):
    return await vk_get_paginated_preview(file_id, page, page_size, db, current_user)