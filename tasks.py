# tasks.py
from celery.exceptions import Ignore
from sqlalchemy import create_engine

import crud
import database
import security_utils
from agents import Orchestrator, SQLCoder, Critic, Storyteller
from celery_worker import celery_app


@celery_app.task(bind=True, time_limit=3600, name='tasks.generate_advanced_report')
def generate_advanced_report(self, connection_id: int, user_id: int, report_id: int):
    """Основная Celery-задача для запуска продвинутого итеративного анализа."""
    db_session = next(database.get_db())
    try:
        self.update_state(state='SETUP', meta={'progress': 'Инициализация анализа...'})
        connection_string = crud.get_decrypted_connection_string(db_session, connection_id, user_id)
        if not connection_string:
            crud.update_report(db_session, report_id, "FAILED", {"error": "Подключение к БД не найдено."})
            raise Ignore()

        engine = create_engine(connection_string, connect_args={'connect_timeout': 20})

        # Инициализация команды агентов
        orchestrator = Orchestrator(engine)
        sql_coder = SQLCoder(engine)
        critic = Critic()
        storyteller = Storyteller()

        self.update_state(state='PLANNING', meta={'progress': 'Генерация начального плана...'})
        analysis_plan = orchestrator.create_initial_plan()
        session_memory = []

        total_tasks = len(analysis_plan)
        completed_tasks = 0
        max_iterations = 20
        current_iteration = 0

        while analysis_plan and current_iteration < max_iterations:
            current_question = analysis_plan.pop(0)
            current_iteration += 1

            progress_percent = int((completed_tasks / total_tasks) * 100) if total_tasks > 0 else 0
            self.update_state(
                state='ANALYZING',
                meta={'progress': f"Шаг {completed_tasks + 1}/{total_tasks}: {current_question}",
                      'percent': progress_percent}
            )

            execution_result = sql_coder.run(current_question)
            critic_evaluation = critic.evaluate(execution_result)

            if critic_evaluation['is_success']:
                orchestrator.process_evaluation(critic_evaluation, session_memory, analysis_plan)
                completed_tasks += 1
                total_tasks = completed_tasks + len(analysis_plan)
            else:
                session_memory.append(critic_evaluation['finding'])
                completed_tasks += 1

        self.update_state(state='SYNTHESIZING', meta={'progress': 'Формирование финального отчета...', 'percent': 99})
        if not session_memory:
            final_report = {"executive_summary": "Анализ не дал результатов.", "detailed_findings": []}
        else:
            final_report = storyteller.narrate(session_memory)

        crud.update_report(db_session, report_id, "COMPLETED", final_report)
        self.update_state(state='SUCCESS', meta={'progress': 'Готово!', 'percent': 100})

    except Exception as e:
        print(f"FATAL error in advanced report task for report_id {report_id}: {e}")
        crud.update_report(db_session, report_id, "FAILED",
                           {"error": "Произошла критическая ошибка.", "details": str(e)})
        raise e
    finally:
        db_session.close()