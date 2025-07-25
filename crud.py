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


def create_database_connection(db: Session, user_id: int, connection_string: str, db_type: str, alias: str) -> models.DatabaseConnection:
    """
    Создает запись о подключении к базе данных, если такой еще не существует.
    """
    # Проверяем, существует ли уже такое подключение
    existing_connection = db.query(models.DatabaseConnection).filter(
        models.DatabaseConnection.user_id == user_id,
        models.DatabaseConnection.connection_string == connection_string
    ).first()

    if existing_connection:
        return existing_connection

    # Если не существует, создаем новое
    db_connection = models.DatabaseConnection(
        user_id=user_id,
        connection_string=connection_string,
        db_type=db_type,
        alias=alias
    )
    db.add(db_connection)
    db.commit()
    db.refresh(db_connection)
    return db_connection


def get_database_connections_by_user_id(db: Session, user_id: int):
    """Получает список подключений пользователя."""
    return db.query(models.DatabaseConnection).filter(models.DatabaseConnection.user_id == user_id).order_by(models.DatabaseConnection.created_at.desc()).all()


def get_report_by_id(db: Session, report_id: int):
    """Получает отчет по ID."""
    return db.query(models.Report).filter(models.Report.id == report_id).first()


def get_reports_by_user_id(db: Session, user_id: int):
    """Получает список отчетов пользователя."""
    return db.query(models.Report).filter(models.Report.user_id == user_id).order_by(models.Report.created_at.desc()).all()


def create_report(db: Session, user_id: int, status: str = "pending") -> models.Report:
    """Создает пустую запись отчета и возвращает ее."""
    report = models.Report(user_id=user_id, status=status)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report

def update_report_status(db: Session, report_id: int, status: str, error_message: str = None):
    """Обновляет статус (и опционально ошибку) отчета."""
    report = db.query(models.Report).filter(models.Report.id == report_id).first()
    if report:
        report.status = status
        if error_message:
            report.results = {"error": error_message} # Записываем ошибку в results
        db.commit()

def update_report_results(db: Session, report_id: int, results: dict, status: str):
    """Обновляет результаты и статус отчета."""
    report = db.query(models.Report).filter(models.Report.id == report_id).first()
    if report:
        report.results = results
        report.status = status
        db.commit()