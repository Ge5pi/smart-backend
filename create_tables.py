# create_tables.py

from database import engine, Base
import models  # Импортируем, чтобы SQLAlchemy "увидел" ваши модели User и File

print("Connecting to the database and creating tables...")

# Эта команда берет все модели, которые наследуются от Base,
# и создает для них соответствующие таблицы в базе данных,
# к которой подключен 'engine'.
Base.metadata.create_all(bind=engine)

print("Tables created successfully!")