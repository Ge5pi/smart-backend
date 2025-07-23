from datetime import datetime, timezone
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional, List
import security_utils
import models
import schemas

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_files_by_user_id(db: Session, user_id: int):
    return db.query(models.File).filter(models.File.user_id == user_id).order_by(
        models.File.datetime_created.desc()).all()


def get_all_files(db: Session):
    return db.query(models.File).all()


def create_user_file(db: Session, user_id: int, file_uid: str, file_name: str, s3_path: str) -> models.File:
    db_file = models.File(
        user_id=user_id,
        file_uid=file_uid,
        file_name=file_name,
        s3_path=s3_path,
        datetime_created=datetime.now(timezone.utc)
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file


def get_file_by_uid(db: Session, file_uid: str) -> models.File:
    return db.query(models.File).filter(models.File.file_uid == file_uid).first()


def create_database_connection(db: Session, user_id: int, connection_string: str, db_type: str):
    """Создает запись о подключении к базе данных."""
    db_connection = models.DatabaseConnection(
        user_id=user_id, connection_string=connection_string, db_type=db_type
    )
    db.add(db_connection)
    db.commit()
    db.refresh(db_connection)
    return db_connection

def get_database_connections_by_user_id(db: Session, user_id: int):
    """Получает список подключений пользователя."""
    return db.query(models.DatabaseConnection).filter(models.DatabaseConnection.user_id == user_id).all()

def get_report_by_id(db: Session, report_id: int):
    """Получает отчет по ID."""
    print("Report_id", report_id)
    return db.query(models.Report).filter(models.Report.id == report_id).first()

def get_reports_by_user_id(db: Session, user_id: int):
    """Получает список отчетов пользователя."""
    return db.query(models.Report).filter(models.Report.user_id == user_id).order_by(models.Report.created_at.desc()).all()
