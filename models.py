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
    connections = relationship("DatabaseConnection", back_populates="user")
    reports = relationship("Report", back_populates="user")


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
    __tablename__ = "db_connections"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    connection_string = Column(String, nullable=False)
    db_type = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="connections")


class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    results = Column(JSON, nullable=True)
    user = relationship("User", back_populates="reports")
