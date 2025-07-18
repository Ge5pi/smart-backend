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


def create_db_connection(db: Session, user_id: int,
                         conn_details: schemas.DatabaseConnectionCreate) -> models.DatabaseConnection:
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
    return db.query(models.DatabaseConnection).filter(
        models.DatabaseConnection.id == connection_id,
        models.DatabaseConnection.user_id == user_id
    ).first()


def get_db_connections_by_user(db: Session, user_id: int) -> list[models.DatabaseConnection]:
    return db.query(models.DatabaseConnection).filter(models.DatabaseConnection.user_id == user_id).all()


def get_decrypted_connection_string(db: Session, connection_id: int, user_id: int) -> str | None:
    db_conn = get_db_connection_by_id(db, connection_id, user_id)
    if db_conn:
        return security_utils.decrypt_data(db_conn.encrypted_connection_string)
    return None


def create_report(db: Session, user_id: int, connection_id: int, task_id: str = None) -> models.Report:
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
    return db.query(models.Report).filter(
        models.Report.id == report_id,
        models.Report.user_id == user_id
    ).first()


def update_report(db: Session, report_id: int, status: str, results: dict):
    db_report = db.query(models.Report).filter(models.Report.id == report_id).first()
    if db_report:
        db_report.status = status
        db_report.results = results
        db.commit()
        db.refresh(db_report)
    return db_report


def get_user_reports(
        db: Session,
        user_id: int,
        limit: int = 10,
        offset: int = 0,
        status_filter: Optional[str] = None
) -> List[models.Report]:
    """
    Получает отчеты пользователя с пагинацией и фильтрацией по статусу.

    Args:
        db: Сессия базы данных
        user_id: ID пользователя
        limit: Максимальное количество отчетов
        offset: Смещение для пагинации
        status_filter: Фильтр по статусу (PENDING, COMPLETED, FAILED)

    Returns:
        Список отчетов пользователя
    """
    query = db.query(models.Report).filter(models.Report.user_id == user_id)

    # Применяем фильтр по статусу если указан
    if status_filter:
        query = query.filter(models.Report.status == status_filter)

    # Сортируем по дате создания (новые первыми)
    query = query.order_by(desc(models.Report.created_at))

    # Применяем пагинацию
    return query.offset(offset).limit(limit).all()


def create_report_feedback(
        db: Session,
        report_id: int,
        user_id: int,
        feedback_data: dict
) -> models.Feedback:
    """
    Создает отзыв для отчета.

    Args:
        db: Сессия базы данных
        report_id: ID отчета
        user_id: ID пользователя
        feedback_data: Данные обратной связи с полями:
            - rating: int (1-5)
            - comment: str (опционально)
            - useful_sections: list[str] (опционально)

    Returns:
        Созданный объект Feedback
    """
    # Проверяем, что отчет существует и принадлежит пользователю
    report = get_report_by_id(db, report_id, user_id)
    if not report:
        raise ValueError("Отчет не найден или не принадлежит пользователю")

    # Проверяем, что пользователь еще не оставлял отзыв на этот отчет
    existing_feedback = db.query(models.Feedback).filter(
        models.Feedback.report_id == report_id,
        models.Feedback.user_id == user_id
    ).first()

    if existing_feedback:
        # Обновляем существующий отзыв
        existing_feedback.rating = feedback_data.get("rating", existing_feedback.rating)
        existing_feedback.comment = feedback_data.get("comment", existing_feedback.comment)
        existing_feedback.useful_sections = feedback_data.get("useful_sections", existing_feedback.useful_sections)
        existing_feedback.created_at = datetime.utcnow()  # Обновляем время
        db.commit()
        db.refresh(existing_feedback)
        return existing_feedback

    # Создаем новый отзыв
    db_feedback = models.Feedback(
        report_id=report_id,
        user_id=user_id,
        rating=feedback_data.get("rating"),
        comment=feedback_data.get("comment"),
        useful_sections=feedback_data.get("useful_sections", [])
    )

    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback


def get_report_feedbacks(db: Session, report_id: int) -> List[models.Feedback]:
    """Получает все отзывы для отчета."""
    return db.query(models.Feedback).filter(models.Feedback.report_id == report_id).all()


def get_user_feedback(db: Session, user_id: int) -> List[models.Feedback]:
    """Получает все отзывы пользователя."""
    return db.query(models.Feedback).filter(models.Feedback.user_id == user_id).order_by(
        desc(models.Feedback.created_at)).all()
