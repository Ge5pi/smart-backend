import hmac, hashlib, base64
from urllib.parse import parse_qsl
from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
import crud
import models
from database import get_db
from urllib.parse import unquote
from config import VK_SECURE_KEY


def get_or_create_vk_user(db: Session, vk_user_id: int) -> models.User:
    user = db.query(models.User).filter(models.User.vk_id == vk_user_id).first()
    if not user:
        user = models.User(
            vk_id=vk_user_id,
            email=f"vk_user_{vk_user_id}@soda.contact",  # Уникальный email-заглушка
            hashed_password="",  # Пароль не нужен
            is_verified=True,  # Считаем пользователя из VK подтвержденным
            is_active=False  # Бесплатный тариф по умолчанию
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def verify_vk_launch_params(request: Request, db: Session = Depends(get_db)) -> models.User:
    print("\n--- DEBUG: НАЧАЛО ПРОВЕРКИ ПОДПИСИ VK ---")

    vk_params_string = request.headers.get("X-VK-Launch-Params")
    if not vk_params_string:
        raise HTTPException(status_code=401, detail="VK launch params not provided")

    print(f"1. Получена строка параметров из заголовка: {vk_params_string}")

    # --- ИЗМЕНЕНИЕ: Более надежный ручной парсинг вместо parse_qsl ---
    query_string = vk_params_string[1:] if vk_params_string.startswith('?') else vk_params_string

    # Сначала разбиваем по '&', потом по '=', и декодируем ключ и значение
    params = {unquote(k): unquote(v) for k, v in (pair.split('=', 1) for pair in query_string.split('&'))}

    print(f"2. Распарсенные и декодированные параметры: {params}")

    vk_user_id = params.get("vk_user_id")
    sign = params.get("sign")

    if not vk_user_id or not sign:
        raise HTTPException(status_code=401, detail="Missing vk_user_id or sign")

    vk_params = {k: v for k, v in params.items() if k.startswith('vk_')}
    print(f"3. Параметры, отфильтрованные для подписи: {vk_params}")

    ordered_params = sorted(vk_params.items())
    sign_payload = "&".join([f"{k}={v}" for k, v in ordered_params])
    print(f"4. Строка, которая будет хэшироваться: '{sign_payload}'")

    digest = hmac.new(VK_SECURE_KEY.encode(), sign_payload.encode(), hashlib.sha256).digest()
    calculated_sign = base64.urlsafe_b64encode(digest).rstrip(b'=').decode()

    print(f"5. Секретный ключ используется (первые 5 символов): '{VK_SECURE_KEY[:5]}...'")
    print(f"6. Подпись, рассчитанная сервером: '{calculated_sign}'")
    print(f"7. Подпись, полученная от VK:     '{sign}'")

    if calculated_sign != sign:
        print("--- DEBUG: ОШИБКА! Подписи НЕ СОВПАДАЮТ. ---")
        raise HTTPException(status_code=401, detail="Invalid signature")

    print("--- DEBUG: УСПЕХ! Подписи совпали. ---")
    return get_or_create_vk_user(db, int(vk_user_id))
