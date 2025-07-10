# models.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
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