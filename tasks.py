# tasks.py
import time
from celery_worker import celery_app
from database import SessionLocal
import crud


# bind=True позволяет получить доступ к самой задаче (self)
# внутри функции. Это нужно, чтобы обновлять ее статус.
@celery_app.task(bind=True)
def generate_full_report(self, connection_id: int, user_id: int, report_id: int):
    """
    Основная задача для генерации отчета.
    Пока что она просто имитирует работу и обновляет статус.
    """
    db = SessionLocal()
    try:
        print(f"Запуск генерации отчета для connection_id: {connection_id}")

        # Обновляем статус в Celery, чтобы фронтенд видел, что работа началась
        self.update_state(state='IN_PROGRESS', meta={'status': 'Анализ начат...'})

        # Имитация долгой работы
        time.sleep(15)

        # Имитация успешного завершения
        final_content = {"summary": "Отчет успешно сгенерирован (это тестовые данные)."}
        crud.update_report(db, report_id=report_id, status="COMPLETED", content=final_content)

        print(f"Генерация отчета {report_id} завершена.")
        return {'status': 'COMPLETED', 'report_id': report_id}

    except Exception as e:
        # В случае ошибки обновляем статус на FAILED
        crud.update_report(db, report_id=report_id, status="FAILED", content={"error": str(e)})
        # Celery автоматически перехватит исключение и пометит задачу как FAILED
        raise e
    finally:
        db.close()