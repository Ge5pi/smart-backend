import io
import json
import uuid
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
import config
import crud
import models


def get_df_from_s3(db: Session, file_id: str) -> pd.DataFrame:
    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="Запись о файле не найдена в БД.")
    try:
        blob = config.gcs_bucket.blob(file_record.s3_path)
        file_content = blob.download_as_bytes()
        file_extension = Path(file_record.file_name).suffix.lower()
        if file_extension == '.csv':
            return pd.read_csv(io.StringIO(file_content.decode("utf-8")))
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(io.BytesIO(file_content))
        else:
            raise HTTPException(status_code=400, detail="Неподдерживаемый тип файла в хранилище.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось прочитать файл из хранилища: {str(e)}")


def save_df_to_s3(db: Session, file_id: str, user_id: int, df: pd.DataFrame):
    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record or file_record.user_id != user_id:
        raise HTTPException(status_code=404, detail="Файл не найден или доступ для сохранения запрещен.")
    try:
        with io.StringIO() as csv_buffer:
            df.to_csv(csv_buffer, index=False)
            blob = config.gcs_bucket.blob(file_record.s3_path)
            blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось сохранить обработанный файл: {str(e)}")


async def logic_upload_file(file: UploadFile, db: Session, current_user: models.User):
    file_id = str(uuid.uuid4())
    original_filename = file.filename
    file_extension = Path(original_filename).suffix.lower()
    if file_extension not in ['.csv', '.xlsx', '.xls']:
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип файла. Допускаются только CSV, XLSX, XLS.")
    s3_path = f"uploads/{current_user.id}/{file_id}/{original_filename}"
    try:
        contents = await file.read()
        blob = config.gcs_bucket.blob(s3_path)
        blob.upload_from_string(contents, content_type=file.content_type)
        db_file = crud.create_user_file(db=db, user_id=current_user.id, file_uid=file_id, file_name=original_filename,
                                        s3_path=s3_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке файла: {e}")
    return db_file


def logic_read_user_files(db: Session, current_user: models.User):
    return crud.get_files_by_user_id(db=db, user_id=current_user.id)


def logic_analyze_existing_file(file_id: str, db: Session):
    df = get_df_from_s3(db, file_id)
    analysis = [{"column": col, "dtype": str(df[col].dtype), "nulls": int(df[col].isna().sum()),
                 "unique": int(df[col].nunique())} for col in df.columns]
    preview_data = df.head(50).fillna("null").to_dict(orient="records")
    return {"columns": analysis, "preview": preview_data, "total_rows": len(df)}


def logic_get_paginated_preview(file_id: str, page: int, page_size: int, db: Session):
    df = get_df_from_s3(db, file_id)
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    paginated_df = df.iloc[start_index:end_index]
    return {"preview": paginated_df.fillna("null").to_dict(orient="records"), "total_rows": len(df)}


def logic_impute_missing(file_id: str, columns_str: str, db: Session, current_user: models.User):
    from main import impute_with_sklearn  # Избегаем циклического импорта
    df = get_df_from_s3(db, file_id)
    selected_columns = json.loads(columns_str)
    df_imputed, _ = impute_with_sklearn(df, selected_columns)
    save_df_to_s3(db, file_id, current_user.id, df_imputed)
    return {"preview": df_imputed.head(50).fillna("null").to_dict(orient="records"), "total_rows": len(df_imputed)}


def logic_detect_outliers(file_id: str, columns_str: str, db: Session):
    from sklearn.ensemble import IsolationForest
    df = get_df_from_s3(db, file_id)
    selected_columns = json.loads(columns_str)
    numeric_df = df[selected_columns].select_dtypes(include=np.number).dropna()
    if numeric_df.empty:
        return {"outlier_count": 0, "outlier_preview": []}
    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(numeric_df)
    outlier_indices = numeric_df.index[predictions == -1]
    outliers_df = df.loc[outlier_indices]
    return {"outlier_count": len(outliers_df),
            "outlier_preview": outliers_df.head(100).fillna("null").to_dict('records')}


def logic_encode_categorical(file_id: str, columns_str: str, db: Session, current_user: models.User):
    df = get_df_from_s3(db, file_id)
    selected_columns = json.loads(columns_str)
    if not all(col in df.columns for col in selected_columns):
        raise HTTPException(status_code=404, detail="Один или несколько столбцов не найдены.")
    df_encoded = pd.get_dummies(df, columns=selected_columns, prefix=selected_columns)
    save_df_to_s3(db, file_id, current_user.id, df_encoded)
    new_analysis = [{"column": col, "dtype": str(df_encoded[col].dtype), "nulls": int(df_encoded[col].isna().sum()),
                     "unique": int(df_encoded[col].nunique())} for col in df_encoded.columns]
    return {"message": "Столбцы успешно закодированы.", "columns": new_analysis,
            "preview": df_encoded.head(50).fillna("null").to_dict(orient="records"), "total_rows": len(df_encoded)}


def logic_download_cleaned_file(file_id: str, db: Session, current_user: models.User):
    from starlette.responses import Response
    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record or file_record.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Файл не найден или доступ запрещен.")
    blob = config.gcs_bucket.blob(file_record.s3_path)
    content = blob.download_as_bytes()
    return Response(content=content, media_type='text/csv',
                    headers={"Content-Disposition": f'attachment; filename=\"{file_record.file_name}\"'})
