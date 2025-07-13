from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


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
    files: list[File] = []

    class Config:
        from_attributes = True


class DatabaseConnectionBase(BaseModel):
    nickname: str = Field(..., min_length=1, max_length=100)
    db_type: str


class DatabaseConnectionCreate(DatabaseConnectionBase):
    connection_string: str = Field(..., min_length=10)


class DatabaseConnectionInfo(DatabaseConnectionBase):
    id: int

    class Config:
        from_attributes = True


class ReportBase(BaseModel):
    id: int
    connection_id: int
    status: str
    created_at: datetime


class ReportInfo(ReportBase):
    task_id: Optional[str] = None

    class Config:
        from_attributes = True


class Report(ReportBase):
    content: Optional[dict] = None

    class Config:
        from_attributes = True
