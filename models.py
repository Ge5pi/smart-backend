from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, LargeBinary, JSON, Text
from datetime import datetime
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    files = relationship("File", back_populates="owner")

class File(Base):
    __tablename__ = "an_files"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    file_uid = Column(String, unique=True, index=True, nullable=False)
    datetime_created = Column(DateTime, nullable=False)
    file_name = Column(String, nullable=False)
    owner = relationship("User", back_populates="files")
    s3_path = Column(String, unique=True, nullable=False)

class DatabaseConnection(Base):
    __tablename__ = "database_connections"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    nickname = Column(String, index=True)
    db_type = Column(String)
    encrypted_connection_string = Column(LargeBinary, nullable=False)
    user = relationship("User")

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    connection_id = Column(Integer, ForeignKey("database_connections.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    task_id = Column(String, unique=True, index=True, nullable=True)
    status = Column(String, default="PENDING")
    content = Column(JSON, nullable=True)  # Используем content вместо results
    created_at = Column(DateTime, default=datetime.utcnow)
    connection = relationship("DatabaseConnection")
    user = relationship("User")
    feedbacks = relationship("Feedback", back_populates="report")

class Feedback(Base):
    __tablename__ = "report_feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5
    comment = Column(Text, nullable=True)
    useful_sections = Column(JSON, nullable=True)  # Список полезных разделов
    created_at = Column(DateTime, default=datetime.utcnow)
    report = relationship("Report", back_populates="feedbacks")
    user = relationship("User")
