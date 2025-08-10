# models.py

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, JSON, Text
from datetime import datetime
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    messages_used = Column(Integer, default=0)
    reports_used = Column(Integer, default=0)
    files = relationship("File", back_populates="owner")
    connections = relationship("DatabaseConnection", back_populates="user")
    reports = relationship("Report", back_populates="user")
    # Добавлено: Связь с сессиями чата
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class File(Base):
    __tablename__ = "an_files"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    file_uid = Column(String, unique=True, index=True, nullable=False)
    datetime_created = Column(DateTime, nullable=False)
    file_name = Column(String, nullable=False)
    owner = relationship("User", back_populates="files")
    s3_path = Column(String, unique=True, nullable=False)
    # Добавлено: Связь с сессиями чата
    chat_sessions = relationship("ChatSession", back_populates="file", cascade="all, delete-orphan")


# Новая модель для сессий чата
class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, index=True)  # UUID сессии
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    file_id = Column(Integer, ForeignKey("an_files.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chat_sessions")
    file = relationship("File", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


# Новая модель для сообщений в чате
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user' или 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


class DatabaseConnection(Base):
    __tablename__ = "db_connections"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    connection_string = Column(String, nullable=False)
    db_type = Column(String, nullable=False)
    alias = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="connections")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    connection_id = Column(Integer, ForeignKey("db_connections.id"), nullable=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    results = Column(JSON, nullable=True)

    user = relationship("User", back_populates="reports")
    connection = relationship("DatabaseConnection")


class EmailVerificationCode(Base):
    __tablename__ = "email_verification_codes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    code = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class SubscriptionOrder(Base):
    __tablename__ = "subscription_orders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    customer_name = Column(String, nullable=False)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")