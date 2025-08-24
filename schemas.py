from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional


# --- Схемы для Файлов ---
class FileBase(BaseModel):
    file_uid: str
    file_name: str


class FileCreate(FileBase):
    pass


class File(FileBase):
    id: int
    user_id: int
    datetime_created: datetime

    class Config:
        from_attributes = True


# --- Схемы для Пользователей ---
class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    is_active: bool
    is_verified: bool

    # Add the expiration_date field
    expiration_date: Optional[datetime] = None

    files: list[File] = []
    messages_used: int
    reports_used: int

    class Config:
        from_attributes = True


# --- Схемы для Чатов ---
class ChatMessageBase(BaseModel):
    role: str
    content: str


class ChatMessage(ChatMessageBase):
    id: int
    session_id: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionInfo(BaseModel):
    id: str
    file_id: int
    created_at: datetime
    last_updated: Optional[datetime] = None
    title: Optional[str] = None

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    history: List[ChatMessage]


class SessionCreate(BaseModel):
    file_id: str


# --- Схемы для Аутентификации и Авторизации ---
class EmailVerification(BaseModel):
    email: str
    code: str


class PasswordResetRequest(BaseModel):
    email: str


class PasswordReset(BaseModel):
    token: str
    new_password: str


# --- Схемы для Подключений к БД и Отчетов ---
class DatabaseConnection(BaseModel):
    id: int
    user_id: int
    connection_string: str
    db_type: str
    alias: str
    created_at: datetime

    class Config:
        from_attributes = True


class Report(BaseModel):
    id: int
    user_id: int
    connection_id: int | None
    status: str
    created_at: datetime
    results: dict | None = None

    class Config:
        from_attributes = True


# --- Схемы для Заказов Подписки ---
class SubscriptionOrderBase(BaseModel):
    customer_name: str


class SubscriptionOrderCreate(SubscriptionOrderBase):
    pass


class SubscriptionOrder(SubscriptionOrderBase):
    id: int
    user_id: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True