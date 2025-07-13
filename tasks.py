# tasks.py
from .celery_worker import celery_app
from .database import SessionLocal
import crud
from sqlalchemy import create_engine
import openai
from services import agents  # Импортируем наших агентов
import config

# Инициализируем OpenAI клиент один раз
client = openai.OpenAI(api_key=config.API_KEY)


@celery_app.task(bind=True)
def generate_full_report(self, connection_id: int, user_id: int, report_id: int):
    """
    Основная задача для генерации отчета.
    Выполняет полный цикл анализа и сохраняет результат в БД.
    """
    db = SessionLocal()
    try:
        # 1. Получение подключения к БД
        self.update_state(state='IN_PROGRESS', meta={'status': 'Подключаюсь к базе данных...'})
        conn_string = crud.get_decrypted_connection_string(db, connection_id, user_id)
        if not conn_string:
            raise ValueError("Не удалось получить доступ к базе данных.")
        engine = create_engine(conn_string)

        # 2. Анализ схемы
        self.update_state(state='IN_PROGRESS', meta={'status': 'Анализирую структуру данных...'})
        schema = agents.get_schema_details(engine)
        if not schema or schema == "No tables found.":
            raise ValueError("В базе данных не найдено таблиц для анализа.")

        # 3. Создание плана анализа
        self.update_state(state='IN_PROGRESS', meta={'status': 'Составляю план анализа...'})
        plan = agents.create_analysis_plan(schema, client)

        # 4. Выполнение плана
        report_sections = []
        total_steps = len(plan)
        for i, question in enumerate(plan):
            # Обновляем статус для фронтенда
            self.update_state(state='IN_PROGRESS', meta={'status': f'Шаг {i + 1}/{total_steps}: {question}'})

            # Выполняем запросы и генерируем контент
            df = agents.run_sql_query_agent(engine, question)
            chart_url = agents.create_visualization(df, question)
            narrative = agents.create_narrative(question, df, chart_url, client)

            # Собираем секцию для отчета
            report_sections.append({
                "title": question,
                "narrative": narrative,
                "chart_url": chart_url,
                "data_preview": df.head(5).to_dict(orient='records') if not df.empty else []
            })

        # 5. Формирование и сохранение финального отчета
        self.update_state(state='IN_PROGRESS', meta={'status': 'Формирую финальный отчет...'})
        final_report_content = {
            "title": f"Аналитический отчет по подключению #{connection_id}",
            "sections": report_sections
        }
        crud.update_report(db, report_id=report_id, status="COMPLETED", content=final_report_content)

        # Возвращаем финальный статус
        return {'status': 'COMPLETED', 'report_id': report_id}

    except Exception as e:
        # В случае любой ошибки логируем ее и обновляем статус
        print(f"Критическая ошибка в задаче генерации отчета (report_id: {report_id}): {e}")
        error_message = f"Произошла ошибка: {str(e)}"
        crud.update_report(db, report_id=report_id, status="FAILED", content={"error": error_message})
        # Передаем исключение в Celery, чтобы он тоже пометил задачу как FAILED
        # и фронтенд получил корректный статус
        self.update_state(state='FAILURE', meta={'status': error_message})
        raise e
    finally:
        # Гарантированно закрываем сессию с БД
        db.close()