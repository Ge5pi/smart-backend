# security_utils.py
from cryptography.fernet import Fernet
import config

try:
    fernet = Fernet(config.ENCRYPTION_KEY)
except Exception as e:
    raise ValueError("Не удалось инициализировать Fernet. Проверьте ваш ENCRYPTION_KEY в config.py") from e


def encrypt_data(data: str) -> bytes:
    """Шифрует строку и возвращает байты."""
    return fernet.encrypt(data.encode('utf-8'))


def decrypt_data(encrypted_data: bytes) -> str:
    """Дешифрует байты и возвращает строку."""
    return fernet.decrypt(encrypted_data).decode('utf-8')
