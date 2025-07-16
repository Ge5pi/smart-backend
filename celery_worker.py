# celery_worker.py
from celery import Celery
import config
import ssl

# Создаем экземпляр Celery
# "tasks" - это стандартное имя для основного модуля с задачами
celery_app = Celery(
    "tasks",
    # Брокер (Broker) - это очередь, куда FastAPI будет класть задачи.
    # Celery worker будет забирать их отсюда.
    broker=config.REDIS_URL,
    # Бэкенд (Backend) - это хранилище, куда Celery worker будет сохранять
    # статусы и результаты выполнения задач.
    backend=config.REDIS_URL,
    # Явно указываем Celery, где искать файл с нашими задачами.
    include=["tasks"],  # <-- Убедитесь, что здесь указан ваш новый файл с задачами
)

# Опциональная конфигурация для улучшения работы
celery_app.conf.update(
    task_track_started=True,
    # Указываем Celery, как обрабатывать SSL для брокера
    broker_transport_options={
        'ssl_cert_reqs': ssl.CERT_REQUIRED
    },
    # Указываем Celery, как обрабатывать SSL для бэкенда результатов
    result_backend_transport_options={
        'ssl_cert_reqs': ssl.CERT_REQUIRED
    }
)