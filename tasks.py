# tasks.py
from celery.exceptions import Ignore
from sqlalchemy import create_engine

import crud
import database
from services.report_agents import Orchestrator, SQLCoder, Critic, Storyteller
from celery_worker import celery_app


@celery_app.task(bind=True, time_limit=3600, name='tasks.generate_advanced_report')
def generate_advanced_report(self, connection_id: int, user_id: int, report_id: int):
    """Основная Celery-задача для запуска продвинутого итеративного анализа."""
    db_session = next(database.get_db())
    try:
        print("\n\n--- [TASK START] ---")
        self.update_state(state='SETUP', meta={'progress': 'Инициализация анализа...'})

        connection_string = crud.get_decrypted_connection_string(db_session, connection_id, user_id)
        if not connection_string:
            crud.update_report(db_session, report_id, "FAILED", {"error": "Подключение к БД не найдено."})
            raise Ignore()
        print("[TASK SETUP] Connection string retrieved successfully.")

        engine = create_engine(connection_string, connect_args={'connect_timeout': 20})

        orchestrator = Orchestrator(engine)
        sql_coder = SQLCoder(engine)
        critic = Critic()
        storyteller = Storyteller()
        print("[TASK SETUP] All agents initialized.")

        self.update_state(state='PLANNING', meta={'progress': 'Генерация интеллектуального плана...'})

        # --- ИЗМЕНЕНИЕ ---
        # Используем новый, более умный метод для создания плана анализа,
        # который анализирует реальную структуру и данные в таблицах.
        analysis_plan = orchestrator.create_intelligent_plan()
        print(f"[TASK PLAN] Intelligent plan received with {len(analysis_plan)} questions: {analysis_plan}")

        session_memory = []
        max_iterations = 20  # Ограничиваем количество итераций для предотвращения бесконечных циклов
        current_iteration = 0

        while analysis_plan and current_iteration < max_iterations:
            current_question = analysis_plan.pop(0)
            current_iteration += 1
            print(f"\n[TASK LOOP {current_iteration}/{max_iterations}] ---> Processing question: '{current_question}'")
            self.update_state(state='ANALYZING', meta={'progress': f"Шаг {current_iteration}: {current_question}"})

            # --- SQL Coder ---
            # SQLCoder теперь имеет встроенную fallback-логику для гарантированного получения данных.
            print("[TASK LOOP] Invoking SQLCoder...")
            execution_result = sql_coder.run(current_question)
            print(
                f"[TASK LOOP] SQLCoder returned. DataFrame rows: {len(execution_result.get('data', pd.DataFrame()))}. Error: {execution_result.get('error')}")

            # --- Critic ---
            # Critic теперь генерирует более детальные инсайты, статистику и новые гипотезы.
            print("[TASK LOOP] Invoking Critic...")
            critic_evaluation = critic.evaluate(execution_result)
            print(
                f"[TASK LOOP] Critic returned. Is success: {critic_evaluation.get('is_success')}. Finding summary: '{critic_evaluation.get('finding', {}).get('summary', 'N/A')[:50]}...'")

            # --- Orchestrator ---
            # Логика обработки результатов не изменилась, но она будет работать
            # с более богатыми данными от Critic.
            if critic_evaluation.get('is_success'):
                print("[TASK LOOP] Invoking Orchestrator to process evaluation.")
                orchestrator.process_evaluation(critic_evaluation, session_memory, analysis_plan)
                print(
                    f"[TASK LOOP] Orchestrator processed. Findings in memory: {len(session_memory)}. New hypotheses added: {len(critic_evaluation.get('new_hypotheses', []))}")
            else:
                session_memory.append(critic_evaluation.get('finding', {"error": "Unknown critic failure"}))
                print("[TASK LOOP] Critic reported failure. Added error finding to memory.")

        # --- Storyteller ---
        self.update_state(state='SYNTHESIZING', meta={'progress': 'Формирование финального отчета...'})
        print(f"\n[TASK FINALIZE] Invoking Storyteller with {len(session_memory)} findings in memory.")
        if not session_memory:
            # Предоставляем более информативное сообщение, если память пуста
            final_report = {
                "executive_summary": "Анализ не дал результатов. Возможные причины: база данных пуста, либо не удалось выполнить ни одного запроса.",
                "detailed_findings": []}
        else:
            # Storyteller теперь создает комплексный отчет с executive summary, метриками и рекомендациями.
            final_report = storyteller.narrate(session_memory)
        print("[TASK FINALIZE] Storyteller finished. Saving report.")

        crud.update_report(db_session, report_id, "COMPLETED", final_report)
        print("[TASK FINALIZE] Report saved successfully. Task complete.")
        self.update_state(state='SUCCESS', meta={'progress': 'Готово!'})

    except Exception as e:
        print(f"--- [TASK FATAL ERROR] ---: {e}")
        # Записываем более детальную информацию об ошибке
        crud.update_report(db_session, report_id, "FAILED",
                           {"error": "Произошла критическая ошибка в задаче.", "details": str(e)})
        # В реальном приложении здесь стоит использовать более надежное логирование
        # import traceback
        # traceback.print_exc()
        raise e
    finally:
        db_session.close()