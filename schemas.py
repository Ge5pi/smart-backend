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
    files: list[File] = []

    class Config:
        from_attributes = True
