from datetime import datetime, timezone

from passlib.context import CryptContext
from sqlalchemy.orm import Session
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
    return db.query(models.File).filter(models.File.user_id == user_id).order_by(models.File.datetime_created.desc()) \
        .all()


def get_all_files(db: Session):
    return db.query(models.File).all()


def create_user_file(db: Session, user_id: int, file_uid: str, file_name: str, s3_path: str) -> models.File:
    db_file = models.File(
        user_id=user_id,
        file_uid=file_uid,
        file_name=file_name,
        s3_path=s3_path,  # <-- Важное поле для облачной версии
        datetime_created=datetime.now(timezone.utc)
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file


def get_file_by_uid(db: Session, file_uid: str) -> models.File:
    return db.query(models.File).filter(models.File.file_uid == file_uid).first()


def create_db_connection(db: Session, user_id: int, conn_details: schemas.DatabaseConnectionCreate) -> models.DatabaseConnection:
    """Создает и сохраняет зашифрованное подключение к БД."""
    encrypted_string = security_utils.encrypt_data(conn_details.connection_string)
    db_conn = models.DatabaseConnection(
        user_id=user_id,
        nickname=conn_details.nickname,
        db_type=conn_details.db_type,
        encrypted_connection_string=encrypted_string
    )
    db.add(db_conn)
    db.commit()
    db.refresh(db_conn)
    return db_conn


def get_db_connection_by_id(db: Session, connection_id: int, user_id: int) -> models.DatabaseConnection | None:
    """Получает подключение по ID, проверяя, что оно принадлежит пользователю."""
    return db.query(models.DatabaseConnection).filter(
        models.DatabaseConnection.id == connection_id,
        models.DatabaseConnection.user_id == user_id
    ).first()


def get_db_connections_by_user(db: Session, user_id: int) -> list[models.DatabaseConnection]:
    """Получает все подключения для указанного пользователя."""
    return db.query(models.DatabaseConnection).filter(models.DatabaseConnection.user_id == user_id).all()


def get_decrypted_connection_string(db: Session, connection_id: int, user_id: int) -> str | None:
    """Безопасно извлекает и дешифрует строку подключения."""
    db_conn = get_db_connection_by_id(db, connection_id, user_id)
    if db_conn:
        return security_utils.decrypt_data(db_conn.encrypted_connection_string)
    return None


def create_report(db: Session, user_id: int, connection_id: int, task_id: str) -> models.Report:
    """Создает начальную запись для нового отчета."""
    db_report = models.Report(
        user_id=user_id,
        connection_id=connection_id,
        task_id=task_id,
        status="PENDING"
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report


def get_report_by_id(db: Session, report_id: int, user_id: int) -> models.Report | None:
    """Получает отчет по ID, проверяя принадлежность пользователю."""
    return db.query(models.Report).filter(
        models.Report.id == report_id,
        models.Report.user_id == user_id
    ).first()


def update_report(db: Session, report_id: int, status: str, content: dict):
    """Обновляет статус и содержимое отчета."""
    db_report = db.query(models.Report).filter(models.Report.id == report_id).first()
    if db_report:
        db_report.status = status
        db_report.content = content
        db.commit()
        db.refresh(db_report)
    return db_report
