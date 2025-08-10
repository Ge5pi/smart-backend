from pydantic import BaseModel
from datetime import datetime


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


class UserBase(BaseModel):
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    id: int
    is_active: bool
    is_verified: bool
    files: list[File] = []
    messages_used: int
    reports_used: int

    class Config:
        from_attributes = True


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


class EmailVerificationRequest(BaseModel):
    email: str


class EmailVerification(BaseModel):
    email: str
    code: str


# Схемы для сброса пароля
class PasswordResetRequest(BaseModel):
    email: str


class PasswordReset(BaseModel):
    token: str
    new_password: str

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
