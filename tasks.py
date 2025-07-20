# tasks.py - полная замена на DataFrame-подход
import logging
from celery.exceptions import Ignore
from sqlalchemy import create_engine
import crud
import database
from services.dataframe_manager import DataFrameManager
from services.dataframe_analyzer import DataFrameAnalyzer
from celery_worker import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, time_limit=7200, name='tasks.generate_dataframe_report')
def generate_dataframe_report(self, connection_id: int, user_id: int, report_id: int, max_questions: int = 15):
    """
    Генерация отчета полностью на основе DataFrame без SQL
    """
    db_session = next(database.get_db())

    try:
        logger.info(f"[DATAFRAME REPORT] Запуск для пользователя {user_id}")

        # === ИНИЦИАЛИЗАЦИЯ ===
        self.update_state(
            state='INITIALIZING',
            meta={'progress': 'Инициализация DataFrame системы...'}
        )

        connection_string = crud.get_decrypted_connection_string(db_session, connection_id, user_id)
        if not connection_string:
            raise ValueError("Подключение не найдено")

        # Создаем подключение только для загрузки данных
        engine = create_engine(connection_string, connect_args={'connect_timeout': 30})

        # === ЗАГРУЗКА ДАННЫХ В DATAFRAME ===
        self.update_state(
            state='LOADING_DATA',
            meta={'progress': 'Загрузка всех таблиц в память...'}
        )

        df_manager = DataFrameManager(engine)
        tables_loaded = df_manager.load_all_tables()

        if not tables_loaded:
            raise ValueError("Не удалось загрузить данные из базы")

        logger.info(f"[DATAFRAME REPORT] Загружено {len(tables_loaded)} таблиц")

        # === СОЗДАНИЕ АНАЛИЗАТОРА ===
        analyzer = DataFrameAnalyzer(df_manager)

        # === СОЗДАНИЕ ПЛАНА АНАЛИЗА ===
        self.update_state(
            state='PLANNING',
            meta={'progress': 'Создание плана анализа на основе DataFrame...'}
        )

        analysis_plan = _create_dataframe_analysis_plan(df_manager, max_questions)

        # === ВЫПОЛНЕНИЕ АНАЛИЗА ===
        session_memory = []

        for i, question in enumerate(analysis_plan):
            if i >= max_questions:
                break

            progress = (i + 1) / min(len(analysis_plan), max_questions) * 100

            self.update_state(
                state='ANALYZING',
                meta={
                    'progress': f'Анализ {i + 1}/{min(len(analysis_plan), max_questions)}: {question}',
                    'progress_percentage': progress
                }
            )

            logger.info(f"[DATAFRAME REPORT] Анализ: {question}")

            # Выполняем анализ без SQL
            result = analyzer.analyze_question(question)

            if not result.get('error'):
                # Готовим данные для сохранения
                data_preview = result.get('data', None)
                if data_preview is not None and hasattr(data_preview, 'head'):
                    data_preview = data_preview.head(10).to_dict('records')
                elif data_preview is None:
                    data_preview = []

                session_memory.append({
                    'question': question,
                    'summary': result.get('summary', ''),
                    'data_preview': data_preview,
                    'chart_data': result.get('chart_data'),
                    'analyzed_tables': result.get('analyzed_tables', []),
                    'method': 'dataframe',
                    'additional_info': {
                        'basic_info': result.get('basic_info'),
                        'numeric_stats': result.get('numeric_stats'),
                        'categorical_stats': result.get('categorical_stats'),
                        'correlations': result.get('correlations'),
                        'anomalies': result.get('anomalies')
                    }
                })
            else:
                logger.error(f"Ошибка анализа вопроса: {result['error']}")
                session_memory.append({
                    'question': question,
                    'summary': f'Ошибка анализа: {result["error"]}',
                    'data_preview': [],
                    'error': result['error'],
                    'method': 'dataframe'
                })

        # === СОЗДАНИЕ ФИНАЛЬНОГО ОТЧЕТА ===
        self.update_state(
            state='FINALIZING',
            meta={'progress': 'Создание финального отчета...'}
        )

        # Получаем сводку по таблицам
        table_summary = df_manager.get_table_summary()

        final_report = {
            "executive_summary": _create_executive_summary(session_memory, table_summary),
            "detailed_findings": session_memory,
            "method": "pure_dataframe",
            "tables_info": table_summary['tables'],
            "relations_info": table_summary['relations'],
            "analysis_stats": {
                "questions_processed": len(session_memory),
                "successful_analyses": len([f for f in session_memory if not f.get('error')]),
                "tables_analyzed": table_summary['total_tables'],
                "relations_found": table_summary['total_relations'],
                "total_memory_mb": round(table_summary['total_memory_mb'], 2)
            },
            "recommendations": _generate_recommendations(session_memory, table_summary)
        }

        crud.update_report(db_session, report_id, "COMPLETED", final_report)
        logger.info(f"[DATAFRAME REPORT] Отчет {report_id} успешно создан")

    except Exception as e:
        logger.error(f"[DATAFRAME REPORT ERROR] {e}", exc_info=True)
        error_report = {
            "error": str(e),
            "method": "dataframe",
            "stage": "error"
        }
        crud.update_report(db_session, report_id, "FAILED", error_report)

    finally:
        db_session.close()


def _create_dataframe_analysis_plan(df_manager: DataFrameManager, max_questions: int) -> list:
    """Создает план анализа на основе загруженных DataFrame"""

    plan = [
        "Общий обзор структуры данных и связей между таблицами"
    ]

    # Добавляем анализ каждой таблицы (ограничиваем количество)
    table_names = list(df_manager.tables.keys())
    for table_name in table_names[:min(5, len(table_names))]:
        plan.append(f"Детальный анализ таблицы '{table_name}' с поиском паттернов и аномалий")

    # Анализ связей если они есть
    if df_manager.relations:
        plan.append("Анализ качества связей между таблицами и их целостности")

    # Специфические анализы
    plan.extend([
        "Агрегированная статистика по ключевым метрикам всех таблиц",
        "Анализ корреляций между числовыми переменными",
        "Поиск аномалий и выбросов в данных",
        "Временной анализ и поиск трендов в данных",
        "Анализ распределения категориальных переменных",
        "Сравнительный анализ таблиц по основным характеристикам"
    ])

    # Если таблиц много, добавляем дополнительные вопросы
    if len(table_names) > 5:
        for table_name in table_names[5:8]:  # Еще несколько таблиц
            plan.append(f"Экспресс-анализ таблицы '{table_name}'")

    return plan[:max_questions]


def _create_executive_summary(session_memory: list, table_summary: dict) -> str:
    """Создает executive summary"""
    successful_analyses = len([f for f in session_memory if not f.get('error')])
    total_questions = len(session_memory)
    total_tables = table_summary['total_tables']
    total_relations = table_summary['total_relations']
    total_memory = table_summary['total_memory_mb']

    # Подсчитываем проанализированные таблицы
    analyzed_tables = set()
    for finding in session_memory:
        analyzed_tables.update(finding.get('analyzed_tables', []))

    summary = (f"Проведен полный DataFrame-анализ базы данных. "
               f"Обработано {successful_analyses} из {total_questions} аналитических вопросов. "
               f"Загружено {total_tables} таблиц ({total_memory:.1f} MB) с {total_relations} связями. "
               f"Проанализировано {len(analyzed_tables)} уникальных таблиц.")

    # Добавляем информацию о найденных инсайтах
    findings_with_anomalies = len([f for f in session_memory
                                   if f.get('additional_info', {}).get('anomalies')])
    findings_with_correlations = len([f for f in session_memory
                                      if f.get('additional_info', {}).get('correlations')])

    if findings_with_anomalies > 0:
        summary += f" Обнаружены аномалии в {findings_with_anomalies} анализах."

    if findings_with_correlations > 0:
        summary += f" Найдены корреляции в {findings_with_correlations} анализах."

    return summary


def _generate_recommendations(session_memory: list, table_summary: dict) -> list:
    """Генерирует рекомендации на основе анализа"""
    recommendations = []

    # Рекомендации по качеству данных
    total_tables = len(table_summary['tables'])
    empty_tables = len([t for t, info in table_summary['tables'].items() if info['rows'] == 0])

    if empty_tables > 0:
        recommendations.append(
            f"Обнаружено {empty_tables} пустых таблиц из {total_tables}. Рекомендуется проверить процессы загрузки данных.")

    # Рекомендации по аномалиям
    findings_with_anomalies = [f for f in session_memory
                               if f.get('additional_info', {}).get('anomalies')]
    if findings_with_anomalies:
        recommendations.append("Найдены аномалии в данных. Рекомендуется настроить мониторинг качества данных.")

    # Рекомендации по связям
    if table_summary['total_relations'] > 0:
        recommendations.append(
            "Обнаружены связи между таблицами. Рекомендуется использовать их для создания комплексных отчетов.")
    else:
        recommendations.append("Связи между таблицами не обнаружены. Рекомендуется проверить настройку внешних ключей.")

    # Рекомендации по производительности
    if table_summary['total_memory_mb'] > 1000:  # Больше 1 GB
        recommendations.append(
            "Большой объем данных в памяти. Рекомендуется рассмотреть оптимизацию загрузки или использование sampling.")

    # Рекомендации по визуализации
    findings_with_charts = len([f for f in session_memory if f.get('chart_data')])
    if findings_with_charts > 0:
        recommendations.append(
            f"Создано {findings_with_charts} визуализаций. Рекомендуется настроить дашборд для регулярного мониторинга.")

    # Общие рекомендации
    recommendations.extend([
        "Автоматизировать регулярное обновление DataFrame-анализа",
        "Создать систему уведомлений о важных изменениях в данных",
        "Настроить мониторинг качества данных на основе найденных паттернов"
    ])

    return recommendations[:8]  # Ограничиваем количество


# Дополнительные задачи для разных типов анализа

@celery_app.task(bind=True, time_limit=3600, name='tasks.quick_dataframe_analysis')
def quick_dataframe_analysis(self, connection_id: int, user_id: int, report_id: int):
    """Быстрый DataFrame анализ (меньше вопросов, быстрее)"""
    return generate_dataframe_report.apply(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 8}
    )


@celery_app.task(bind=True, time_limit=9000, name='tasks.comprehensive_dataframe_analysis')
def comprehensive_dataframe_analysis(self, connection_id: int, user_id: int, report_id: int):
    """Комплексный DataFrame анализ (максимальное количество вопросов)"""
    return generate_dataframe_report.apply(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 25}
    )


# Оставляем старые функции для обратной совместимости
@celery_app.task(bind=True, time_limit=3600, name='tasks.generate_advanced_report')
def generate_advanced_report(self, connection_id: int, user_id: int, report_id: int):
    """Legacy функция - перенаправляет на DataFrame анализ"""
    logger.warning("Используется legacy функция generate_advanced_report, перенаправляем на DataFrame анализ")
    return generate_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 12}
    )


@celery_app.task(bind=True, time_limit=1800, name='tasks.quick_ml_analysis')
def quick_ml_analysis(self, connection_id: int, user_id: int, report_id: int):
    """Legacy функция - перенаправляет на быстрый DataFrame анализ"""
    logger.warning("Используется legacy функция quick_ml_analysis, перенаправляем на DataFrame анализ")
    return quick_dataframe_analysis.apply_async(args=[connection_id, user_id, report_id])
