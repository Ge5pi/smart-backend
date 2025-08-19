import hmac
import hashlib
import json
from urllib.parse import unquote, parse_qsl
from fastapi import Depends, HTTPException, Header
from sqlalchemy.orm import Session
from sqlalchemy import BigInteger # Импортируем BigInteger
import crud
import models
import database
import os

# ВАЖНО: Храните токен в переменных окружения, а не в коде!
# Установите переменную окружения TELEGRAM_BOT_TOKEN
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")

def is_valid_init_data(init_data: str, bot_token: str) -> bool:
    """Проверяет валидность строки initData от Telegram."""
    try:
        parsed_data = dict(parse_qsl(unquote(init_data)))
        received_hash = parsed_data.pop('hash')
        data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(parsed_data.items()))
        secret_key = hmac.new("WebAppData".encode(), bot_token.encode(), hashlib.sha256).digest()
        calculated_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
        return calculated_hash == received_hash
    except (KeyError, ValueError):
        return False

async def verify_telegram_init_data(
    x_telegram_init_data: str = Header(..., alias="X-Telegram-Init-Data"),
    db: Session = Depends(database.get_db)
) -> models.User:
    """
    Зависимость FastAPI для проверки данных запуска Telegram Mini App.
    Ищет пользователя по tg_id или создает нового.
    """
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
         raise HTTPException(status_code=500, detail="Telegram Bot Token не настроен на сервере.")

    if not is_valid_init_data(x_telegram_init_data, BOT_TOKEN):
        raise HTTPException(status_code=403, detail="Невалидные данные от Telegram (Invalid InitData)")

    init_data_dict = dict(parse_qsl(unquote(x_telegram_init_data)))
    user_data_str = init_data_dict.get('user')
    if not user_data_str:
        raise HTTPException(status_code=400, detail="Данные пользователя не найдены в InitData")

    user_data = json.loads(user_data_str)
    tg_id = user_data.get('id')
    if not tg_id:
        raise HTTPException(status_code=400, detail="ID пользователя не найден в InitData")

    # Используем новое универсальное поле platform_id
    user = db.query(models.User).filter(models.User.platform_id == tg_id).first()

    if not user:
        fake_email = f"tg_user_{tg_id}@telegram.org"
        existing_user = crud.get_user_by_email(db, email=fake_email)
        if existing_user:
            user = existing_user
        else:
            new_user_data = models.User(
                email=fake_email,
                platform_id=tg_id,
                is_active=True,
                is_verified=True
            )
            db.add(new_user_data)
            db.commit()
            db.refresh(new_user_data)
            user = new_user_data
    return user