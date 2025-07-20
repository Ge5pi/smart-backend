# tasks.py - полная DataFrame система
import logging
from celery.exceptions import Ignore
from sqlalchemy import create_engine
import crud
import database
from services.dataframe_manager import DataFrameManager
from services.dataframe_analyzer import DataFrameAnalyzer
from utils.json_serializer import convert_to_serializable
from celery_worker import celery_app
from datetime import datetime
import json
import sys
import os

# Добавляем путь для импорта utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, time_limit=7200, name='tasks.generate_dataframe_report')
def generate_dataframe_report(self, connection_id: int, user_id: int, report_id: int, max_questions: int = 15):
    """
    Генерация отчета полностью на основе DataFrame без SQL
    """
    db_session = next(database.get_db())

    try:
        logger.info(f"[DATAFRAME REPORT] Запуск для пользователя {user_id}, отчет {report_id}")

        # === ИНИЦИАЛИЗАЦИЯ ===
        self.update_state(
            state='INITIALIZING',
            meta={'progress': 'Инициализация DataFrame системы...'}
        )

        connection_string = crud.get_decrypted_connection_string(db_session, connection_id, user_id)
        if not connection_string:
            raise ValueError("Подключение не найдено или недоступно")

        # Создаем подключение только для загрузки данных
        engine = create_engine(connection_string, connect_args={'connect_timeout': 60})

        # === ЗАГРУЗКА ДАННЫХ В DATAFRAME ===
        self.update_state(
            state='LOADING_DATA',
            meta={'progress': 'Загрузка всех таблиц в память...', 'progress_percentage': 10}
        )

        df_manager = DataFrameManager(engine)
        tables_loaded = df_manager.load_all_tables(max_rows_per_table=50000)  # Ограничиваем для стабильности

        if not tables_loaded:
            raise ValueError("Не удалось загрузить данные из базы. Проверьте подключение и права доступа.")

        logger.info(f"[DATAFRAME REPORT] Успешно загружено {len(tables_loaded)} таблиц")

        # === ОПТИМИЗАЦИЯ ПАМЯТИ ===
        self.update_state(
            state='OPTIMIZING',
            meta={'progress': 'Оптимизация использования памяти...', 'progress_percentage': 20}
        )

        df_manager.optimize_memory()

        # === СОЗДАНИЕ АНАЛИЗАТОРА ===
        analyzer = DataFrameAnalyzer(df_manager)

        # === СОЗДАНИЕ ПЛАНА АНАЛИЗА ===
        self.update_state(
            state='PLANNING',
            meta={'progress': 'Создание плана анализа на основе DataFrame...', 'progress_percentage': 25}
        )

        analysis_plan = _create_dataframe_analysis_plan(df_manager, max_questions)
        logger.info(f"[DATAFRAME REPORT] План анализа: {len(analysis_plan)} вопросов")

        # === ВЫПОЛНЕНИЕ АНАЛИЗА ===
        session_memory = []
        successful_analyses = 0

        for i, question in enumerate(analysis_plan):
            if i >= max_questions:
                break

            progress = 25 + (i + 1) / min(len(analysis_plan), max_questions) * 65  # 25-90%

            self.update_state(
                state='ANALYZING',
                meta={
                    'progress': f'Анализ {i + 1}/{min(len(analysis_plan), max_questions)}: {question}',
                    'progress_percentage': progress
                }
            )

            logger.info(f"[DATAFRAME REPORT] Анализ {i + 1}: {question}")

            try:
                # Выполняем анализ без SQL
                result = analyzer.analyze_question(question)

                if not result.get('error'):
                    # Безопасная подготовка данных для сериализации
                    data_preview = result.get('data', None)
                    if data_preview:
                        if hasattr(data_preview, 'head'):
                            # Это DataFrame
                            data_preview = convert_to_serializable(data_preview.head(10).to_dict('records'))
                        elif isinstance(data_preview, (list, dict)):
                            # Уже список или словарь
                            data_preview = convert_to_serializable(data_preview)
                    else:
                        data_preview = []

                    # Конвертируем все дополнительные данные
                    additional_info = result.get('additional_info', {})
                    if additional_info:
                        additional_info = convert_to_serializable(additional_info)

                    finding_entry = {
                        'question': str(question),
                        'summary': str(result.get('summary', '')),
                        'data_preview': data_preview,
                        'chart_data': convert_to_serializable(result.get('chart_data')),
                        'analyzed_tables': list(result.get('analyzed_tables', [])),
                        'method': 'dataframe',
                        'additional_info': additional_info,
                        'timestamp': datetime.now().isoformat()
                    }

                    session_memory.append(finding_entry)
                    successful_analyses += 1

                    logger.info(f"[DATAFRAME REPORT] ✅ Анализ {i + 1} завершен успешно")

                else:
                    error_msg = str(result.get('error', 'Неизвестная ошибка'))
                    logger.error(f"[DATAFRAME REPORT] ❌ Ошибка анализа {i + 1}: {error_msg}")

                    session_memory.append({
                        'question': str(question),
                        'summary': f'Ошибка анализа: {error_msg}',
                        'data_preview': [],
                        'error': error_msg,
                        'method': 'dataframe',
                        'timestamp': datetime.now().isoformat()
                    })

            except Exception as analysis_error:
                error_msg = f"Критическая ошибка анализа: {str(analysis_error)}"
                logger.error(f"[DATAFRAME REPORT] 💥 {error_msg}")

                session_memory.append({
                    'question': str(question),
                    'summary': error_msg,
                    'data_preview': [],
                    'error': error_msg,
                    'method': 'dataframe',
                    'timestamp': datetime.now().isoformat()
                })

        # === СОЗДАНИЕ ФИНАЛЬНОГО ОТЧЕТА ===
        self.update_state(
            state='FINALIZING',
            meta={'progress': 'Создание финального отчета...', 'progress_percentage': 90}
        )

        logger.info(
            f"[DATAFRAME REPORT] Создание итогового отчета: {successful_analyses}/{len(session_memory)} успешных анализов")

        # Получаем сводку по таблицам
        table_summary = df_manager.get_table_summary()
        memory_info = df_manager.get_memory_usage()

        # Создаем финальный отчет с безопасной сериализацией
        final_report = {
            "executive_summary": str(_create_executive_summary(session_memory, table_summary, successful_analyses)),
            "detailed_findings": convert_to_serializable(session_memory),
            "method": "pure_dataframe",
            "tables_info": convert_to_serializable(table_summary['tables']),
            "relations_info": convert_to_serializable(table_summary['relations']),
            "analysis_stats": {
                "questions_processed": int(len(session_memory)),
                "successful_analyses": int(successful_analyses),
                "failed_analyses": int(len(session_memory) - successful_analyses),
                "tables_analyzed": int(table_summary['total_tables']),
                "relations_found": int(table_summary['total_relations']),
                "total_memory_mb": round(float(table_summary['total_memory_mb']), 2),
                "analysis_duration_minutes": "N/A",  # Можно добавить если нужно
                "success_rate_percent": round(float(successful_analyses / max(len(session_memory), 1) * 100), 1)
            },
            "memory_usage": convert_to_serializable(memory_info),
            "recommendations": [str(r) for r in
                                _generate_recommendations(session_memory, table_summary, successful_analyses)],
            "report_metadata": {
                "created_at": datetime.now().isoformat(),
                "user_id": int(user_id),
                "connection_id": int(connection_id),
                "report_version": "2.0_dataframe",
                "max_questions_requested": int(max_questions)
            }
        }

        # Проверяем размер отчета перед сохранением
        try:
            report_json = json.dumps(final_report, ensure_ascii=False)
            report_size_mb = len(report_json.encode('utf-8')) / 1024 / 1024
            logger.info(f"[DATAFRAME REPORT] Размер итогового отчета: {report_size_mb:.2f} MB")

            # Если отчет слишком большой, сокращаем данные
            if report_size_mb > 10:  # 10MB лимит
                logger.warning(f"[DATAFRAME REPORT] Отчет слишком большой ({report_size_mb:.2f} MB), сокращаем данные")
                final_report = _trim_large_report(final_report)

        except Exception as json_error:
            logger.error(f"[DATAFRAME REPORT] Ошибка JSON сериализации: {json_error}")
            # Дополнительная очистка
            final_report = convert_to_serializable(final_report)

        # === СОХРАНЕНИЕ ОТЧЕТА ===
        self.update_state(
            state='SAVING',
            meta={'progress': 'Сохранение отчета в базу данных...', 'progress_percentage': 95}
        )

        try:
            crud.update_report(db_session, report_id, "COMPLETED", final_report)
            logger.info(f"[DATAFRAME REPORT] ✅ Отчет {report_id} успешно сохранен")
        except Exception as save_error:
            logger.error(f"[DATAFRAME REPORT] Ошибка сохранения отчета: {save_error}")

            # Откатываем транзакцию
            try:
                db_session.rollback()
            except:
                pass

            # Попытка сохранить упрощенный отчет
            simplified_report = {
                "executive_summary": final_report["executive_summary"],
                "method": "pure_dataframe",
                "analysis_stats": final_report["analysis_stats"],
                "error": f"Полный отчет не удалось сохранить: {str(save_error)}",
                "report_metadata": final_report["report_metadata"]
            }

            crud.update_report(db_session, report_id, "COMPLETED", simplified_report)
            logger.warning(f"[DATAFRAME REPORT] Сохранен упрощенный отчет {report_id}")

        self.update_state(
            state='SUCCESS',
            meta={'progress': 'DataFrame-анализ завершен успешно!', 'progress_percentage': 100}
        )

        return {
            "status": "success",
            "report_id": report_id,
            "questions_processed": len(session_memory),
            "successful_analyses": successful_analyses,
            "tables_loaded": len(tables_loaded)
        }

    except Exception as e:
        logger.error(f"[DATAFRAME REPORT ERROR] Критическая ошибка: {e}", exc_info=True)

        # Откатываем транзакцию
        try:
            db_session.rollback()
        except:
            pass

        error_report = {
            "error": str(e),
            "method": "dataframe",
            "stage": "critical_error",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "connection_id": connection_id
        }

        try:
            crud.update_report(db_session, report_id, "FAILED", error_report)
            logger.info(f"[DATAFRAME REPORT] Ошибка сохранена в отчет {report_id}")
        except Exception as save_error:
            logger.error(f"[DATAFRAME REPORT] Не удалось сохранить ошибку: {save_error}")

        # Уведомляем Celery об ошибке
        self.update_state(
            state='FAILURE',
            meta={
                'progress': f'Критическая ошибка: {str(e)}',
                'error': str(e),
                'progress_percentage': 0
            }
        )

        raise e

    finally:
        try:
            db_session.close()
        except:
            pass


def _create_dataframe_analysis_plan(df_manager: DataFrameManager, max_questions: int) -> List[str]:
    """Создает интеллектуальный план анализа на основе загруженных DataFrame"""

    plan = [
        "Общий обзор структуры данных и связей между таблицами"
    ]

    table_names = list(df_manager.tables.keys())

    # Анализ каждой важной таблицы
    tables_by_size = sorted(df_manager.tables.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (table_name, df) in enumerate(tables_by_size[:min(6, len(tables_by_size))]):
        if i < 3:  # Первые 3 таблицы - детальный анализ
            plan.append(f"Детальный анализ таблицы '{table_name}' с поиском паттернов и аномалий")
        else:  # Остальные - экспресс анализ
            plan.append(f"Экспресс-анализ структуры таблицы '{table_name}'")

    # Анализ связей если они есть
    if df_manager.relations:
        plan.append("Анализ качества связей между таблицами и их целостности")

    # Специфические анализы в зависимости от данных
    has_numeric_data = any(
        len(df.select_dtypes(include=[np.number]).columns) > 0
        for df in df_manager.tables.values()
    )

    has_datetime_data = any(
        any('date' in col.lower() or 'time' in col.lower() for col in df.columns)
        for df in df_manager.tables.values()
    )

    if has_numeric_data:
        plan.extend([
            "Агрегированная статистика по ключевым числовым метрикам",
            "Анализ корреляций между числовыми переменными",
            "Поиск аномалий и выбросов в числовых данных"
        ])

    if has_datetime_data:
        plan.append("Временной анализ и поиск трендов в данных")

    # Общие анализы
    plan.extend([
        "Анализ распределения категориальных переменных",
        "Сравнительный анализ таблиц по основным характеристикам"
    ])

    # Если таблиц много, добавляем дополнительные специфические вопросы
    if len(table_names) > 6:
        plan.append("Поиск дублированных и пропущенных данных во всех таблицах")

        # Анализ таблиц с наибольшим количеством связей
        table_connections = {}
        for relation in df_manager.relations:
            table_connections[relation.from_table] = table_connections.get(relation.from_table, 0) + 1
            table_connections[relation.to_table] = table_connections.get(relation.to_table, 0) + 1

        if table_connections:
            most_connected_table = max(table_connections.items(), key=lambda x: x[1])[0]
            plan.append(f"Углубленный анализ центральной таблицы '{most_connected_table}' и её связей")

    # Продвинутые анализы для больших планов
    if max_questions > 20:
        plan.extend([
            "Анализ качества данных и рекомендации по очистке",
            "Поиск скрытых паттернов и закономерностей",
            "Оценка потенциала для машинного обучения",
            "Рекомендации по оптимизации структуры данных"
        ])

    return plan[:max_questions]


def _create_executive_summary(session_memory: List[dict], table_summary: dict, successful_analyses: int) -> str:
    """Создает executive summary для отчета"""
    try:
        total_questions = len(session_memory)
        total_tables = table_summary.get('total_tables', 0)
        total_relations = table_summary.get('total_relations', 0)
        total_memory = table_summary.get('total_memory_mb', 0)

        # Подсчитываем проанализированные таблицы
        analyzed_tables = set()
        for finding in session_memory:
            analyzed_tables.update(finding.get('analyzed_tables', []))

        success_rate = (successful_analyses / max(total_questions, 1)) * 100

        # Базовая информация
        summary_parts = [
            f"Проведен полный DataFrame-анализ базы данных.",
            f"Обработано {successful_analyses} из {total_questions} аналитических вопросов (успешность: {success_rate:.1f}%).",
            f"Загружено {total_tables} таблиц ({total_memory:.1f} MB) с {total_relations} связями.",
            f"Проанализировано {len(analyzed_tables)} уникальных таблиц."
        ]

        # Анализ найденных инсайтов
        findings_with_anomalies = 0
        findings_with_correlations = 0
        findings_with_trends = 0

        for finding in session_memory:
            additional_info = finding.get('additional_info', {})
            if additional_info:
                if additional_info.get('anomalies'):
                    findings_with_anomalies += 1
                if additional_info.get('correlations'):
                    findings_with_correlations += 1
                if 'тренд' in finding.get('summary', '').lower():
                    findings_with_trends += 1

        # Добавляем информацию о найденных инсайтах
        if findings_with_anomalies > 0:
            summary_parts.append(f"Обнаружены аномалии в {findings_with_anomalies} анализах.")

        if findings_with_correlations > 0:
            summary_parts.append(f"Найдены корреляции в {findings_with_correlations} анализах.")

        if findings_with_trends > 0:
            summary_parts.append(f"Выявлены временные тренды в {findings_with_trends} анализах.")

        # Информация о качестве данных
        tables_info = table_summary.get('tables', {})
        if tables_info:
            total_rows = sum(info.get('rows', 0) for info in tables_info.values())
            avg_columns = sum(info.get('columns', 0) for info in tables_info.values()) / len(tables_info)

            summary_parts.append(
                f"Общий объем данных: {total_rows:,} записей, средне {avg_columns:.1f} колонок на таблицу.")

        # Рекомендации по дальнейшим действиям
        if total_relations > 0:
            summary_parts.append("Рекомендуется использовать обнаруженные связи для создания комплексных отчетов.")

        if findings_with_anomalies > 0:
            summary_parts.append("Необходимо внимание к качеству данных из-за найденных аномалий.")

        return " ".join(summary_parts)

    except Exception as e:
        logger.error(f"Ошибка создания executive summary: {e}")
        return f"DataFrame-анализ завершен с {successful_analyses} успешными анализами из {len(session_memory)} запланированных."


def _generate_recommendations(session_memory: List[dict], table_summary: dict, successful_analyses: int) -> List[str]:
    """Генерирует рекомендации на основе результатов анализа"""
    try:
        recommendations = []

        total_tables = table_summary.get('total_tables', 0)
        total_relations = table_summary.get('total_relations', 0)
        total_memory = table_summary.get('total_memory_mb', 0)
        tables_info = table_summary.get('tables', {})

        # Рекомендации по структуре данных
        if total_tables > 10:
            recommendations.append(
                "Рассмотрите возможность консолидации данных - обнаружено много таблиц, что может усложнить анализ.")

        if total_relations == 0 and total_tables > 1:
            recommendations.append(
                "Критично: связи между таблицами не обнаружены. Проверьте настройку внешних ключей для обеспечения целостности данных.")
        elif total_relations > 0:
            recommendations.append(
                f"Отлично: обнаружено {total_relations} связей между таблицами. Используйте их для создания интегрированных отчетов.")

        # Рекомендации по качеству данных
        findings_with_anomalies = sum(1 for f in session_memory
                                      if f.get('additional_info', {}).get('anomalies'))

        if findings_with_anomalies > 0:
            recommendations.append(
                f"Внимание: найдены аномалии в {findings_with_anomalies} анализах. Настройте систему мониторинга качества данных.")

        # Рекомендации по корреляциям
        findings_with_correlations = sum(1 for f in session_memory
                                         if f.get('additional_info', {}).get('correlations'))

        if findings_with_correlations > 0:
            recommendations.append(
                f"Возможность: обнаружены корреляции в {findings_with_correlations} анализах. Рассмотрите создание предиктивных моделей.")

        # Рекомендации по производительности
        if total_memory > 500:  # Больше 500 MB
            recommendations.append(
                "Оптимизация: большой объем данных в памяти. Рассмотрите использование индексов и партиционирования.")

        if total_memory > 1000:  # Больше 1 GB
            recommendations.append(
                "Критично: очень большой объем данных. Рекомендуется архивирование старых данных и использование сэмплинга для анализа.")

        # Рекомендации по анализу успешности
        success_rate = (successful_analyses / max(len(session_memory), 1)) * 100

        if success_rate < 80:
            recommendations.append(
                f"Внимание: низкая успешность анализа ({success_rate:.1f}%). Проверьте качество и структуру данных.")
        elif success_rate > 95:
            recommendations.append("Отлично: высокая успешность анализа. Данные хорошо структурированы для аналитики.")

        # Рекомендации по визуализации
        findings_with_charts = sum(1 for f in session_memory if f.get('chart_data'))
        if findings_with_charts > 0:
            recommendations.append(
                f"Визуализация: создано {findings_with_charts} графиков. Настройте дашборд для регулярного мониторинга ключевых метрик.")

        # Рекомендации по пропущенным данным
        if tables_info:
            tables_with_issues = []
            for table_name, info in tables_info.items():
                schema_info = info.get('schema_info', {})
                if schema_info.get('is_truncated'):
                    tables_with_issues.append(table_name)

            if tables_with_issues:
                recommendations.append(
                    f"Предупреждение: данные в таблицах {', '.join(tables_with_issues)} были ограничены для анализа. Для полного анализа увеличьте лимиты.")

        # Временные рекомендации
        has_datetime_analysis = any('время' in f.get('summary', '').lower() or 'тренд' in f.get('summary', '').lower()
                                    for f in session_memory)

        if has_datetime_analysis:
            recommendations.append(
                "Временной анализ: обнаружены временные паттерны. Настройте регулярное обновление анализа для отслеживания трендов.")

        # Рекомендации по машинному обучению
        numeric_tables = sum(1 for info in tables_info.values()
                             if info.get('columns', 0) > 3 and info.get('rows', 0) > 100)

        if numeric_tables > 0 and findings_with_correlations > 0:
            recommendations.append(
                "ML возможности: данные подходят для машинного обучения. Рассмотрите создание предиктивных моделей.")

        # Общие рекомендации по автоматизации
        recommendations.extend([
            "Автоматизация: настройте регулярное обновление DataFrame-анализа для мониторинга изменений.",
            "Уведомления: создайте систему алертов на основе обнаруженных аномалий и трендов.",
            "Интеграция: рассмотрите интеграцию результатов анализа с существующими BI-системами."
        ])

        # Ограничиваем количество рекомендаций
        return recommendations[:12]

    except Exception as e:
        logger.error(f"Ошибка генерации рекомендаций: {e}")
        return [
            "Завершен DataFrame-анализ данных",
            "Рекомендуется регулярное обновление анализа",
            "Настройте мониторинг качества данных",
            "Рассмотрите создание автоматических отчетов"
        ]


def _trim_large_report(report: dict) -> dict:
    """Сокращает размер слишком большого отчета"""
    try:
        logger.info("Сокращение размера отчета для экономии памяти")

        # Создаем копию отчета
        trimmed_report = report.copy()

        # Сокращаем detailed_findings
        detailed_findings = trimmed_report.get('detailed_findings', [])
        if len(detailed_findings) > 0:
            for finding in detailed_findings:
                # Ограничиваем размер data_preview
                if isinstance(finding.get('data_preview'), list) and len(finding['data_preview']) > 5:
                    finding['data_preview'] = finding['data_preview'][:5]
                    finding['data_preview'].append(
                        {"note": f"... показано 5 из {len(finding['data_preview'])} записей"})

                # Упрощаем additional_info
                if finding.get('additional_info'):
                    additional_info = finding['additional_info']

                    # Сокращаем аномалии
                    if additional_info.get('anomalies') and len(additional_info['anomalies']) > 3:
                        additional_info['anomalies'] = additional_info['anomalies'][:3]

                    # Сокращаем корреляции
                    if additional_info.get('correlations') and len(additional_info['correlations']) > 5:
                        additional_info['correlations'] = additional_info['correlations'][:5]

                    # Упрощаем numeric_stats
                    if additional_info.get('numeric_stats'):
                        additional_info['numeric_stats'] = {"note": "Числовая статистика сокращена для экономии места"}

        # Сокращаем tables_info
        tables_info = trimmed_report.get('tables_info', {})
        for table_name, info in tables_info.items():
            # Убираем детальную schema_info
            if 'schema_info' in info:
                schema_info = info['schema_info']
                info['schema_info'] = {
                    'row_count': schema_info.get('row_count'),
                    'memory_usage_mb': schema_info.get('memory_usage_mb'),
                    'is_truncated': schema_info.get('is_truncated'),
                    'note': "Детальная схема сокращена"
                }

        # Упрощаем memory_usage
        if 'memory_usage' in trimmed_report:
            trimmed_report['memory_usage'] = {"note": "Детальная информация о памяти сокращена"}

        # Добавляем информацию о сокращении
        trimmed_report['report_metadata']['trimmed'] = True
        trimmed_report['report_metadata']['trim_reason'] = "Отчет сокращен для экономии памяти"

        logger.info("Отчет успешно сокращен")
        return trimmed_report

    except Exception as e:
        logger.error(f"Ошибка сокращения отчета: {e}")
        return report


# =============== ДОПОЛНИТЕЛЬНЫЕ ЗАДАЧИ ===============

@celery_app.task(bind=True, time_limit=3600, name='tasks.quick_dataframe_analysis')
def quick_dataframe_analysis(self, connection_id: int, user_id: int, report_id: int):
    """Быстрый DataFrame анализ (меньше вопросов, быстрее выполнение)"""
    logger.info(f"[QUICK DATAFRAME] Запуск быстрого анализа для пользователя {user_id}")
    return generate_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 8}
    ).get()


@celery_app.task(bind=True, time_limit=9000, name='tasks.comprehensive_dataframe_analysis')
def comprehensive_dataframe_analysis(self, connection_id: int, user_id: int, report_id: int):
    """Комплексный DataFrame анализ (максимальное количество вопросов)"""
    logger.info(f"[COMPREHENSIVE DATAFRAME] Запуск полного анализа для пользователя {user_id}")
    return generate_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 25}
    ).get()


@celery_app.task(bind=True, time_limit=5400, name='tasks.custom_dataframe_analysis')
def custom_dataframe_analysis(self, connection_id: int, user_id: int, report_id: int,
                              max_questions: int = 15, focus_tables: List[str] = None):
    """Настраиваемый DataFrame анализ с фокусом на определенные таблицы"""
    logger.info(f"[CUSTOM DATAFRAME] Запуск настраиваемого анализа для пользователя {user_id}")

    # Здесь можно добавить логику фокусировки на определенных таблицах
    # Пока просто вызываем стандартный анализ
    return generate_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': max_questions}
    ).get()


# =============== LEGACY ПОДДЕРЖКА ===============

@celery_app.task(bind=True, time_limit=3600, name='tasks.generate_advanced_report')
def generate_advanced_report(self, connection_id: int, user_id: int, report_id: int):
    """Legacy функция - перенаправляет на DataFrame анализ"""
    logger.warning(
        f"[LEGACY] Используется устаревшая функция generate_advanced_report для пользователя {user_id}, перенаправляем на DataFrame анализ")

    return generate_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 12}
    ).get()


@celery_app.task(bind=True, time_limit=1800, name='tasks.quick_ml_analysis')
def quick_ml_analysis(self, connection_id: int, user_id: int, report_id: int):
    """Legacy функция - перенаправляет на быстрый DataFrame анализ"""
    logger.warning(
        f"[LEGACY] Используется устаревшая функция quick_ml_analysis для пользователя {user_id}, перенаправляем на DataFrame анализ")

    return quick_dataframe_analysis.apply_async(
        args=[connection_id, user_id, report_id]
    ).get()


# =============== УТИЛИТАРНЫЕ ЗАДАЧИ ===============

@celery_app.task(bind=True, time_limit=600, name='tasks.test_dataframe_connection')
def test_dataframe_connection(self, connection_id: int, user_id: int):
    """Тестирует подключение и возвращает базовую информацию о данных"""

    db_session = next(database.get_db())

    try:
        logger.info(f"[TEST CONNECTION] Тестирование подключения {connection_id} для пользователя {user_id}")

        connection_string = crud.get_decrypted_connection_string(db_session, connection_id, user_id)
        if not connection_string:
            raise ValueError("Подключение не найдено")

        engine = create_engine(connection_string, connect_args={'connect_timeout': 30})

        # Создаем быстрый тест DataFrameManager
        df_manager = DataFrameManager(engine)

        # Загружаем только первые несколько строк из каждой таблицы
        from sqlalchemy import inspect
        inspector = inspect(engine)

        test_results = {
            "connection_status": "success",
            "tables_found": [],
            "total_tables": 0,
            "estimated_total_rows": 0,
            "sample_data_available": True
        }

        table_names = [t for t in inspector.get_table_names() if t != 'alembic_version']
        test_results["total_tables"] = len(table_names)

        for table_name in table_names[:10]:  # Тестируем максимум 10 таблиц
            try:
                with engine.connect() as conn:
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.scalar() or 0

                    sample_query = text(f"SELECT * FROM {table_name} LIMIT 1")
                    sample_result = conn.execute(sample_query)
                    columns = list(sample_result.keys())

                    test_results["tables_found"].append({
                        "table_name": table_name,
                        "row_count": int(row_count),
                        "columns": len(columns),
                        "column_names": columns[:10]  # Первые 10 колонок
                    })

                    test_results["estimated_total_rows"] += row_count

            except Exception as table_error:
                logger.warning(f"Ошибка тестирования таблицы {table_name}: {table_error}")
                test_results["tables_found"].append({
                    "table_name": table_name,
                    "error": str(table_error)
                })

        # Проверяем связи
        relations_found = 0
        for table_name in table_names[:5]:  # Проверяем связи для первых 5 таблиц
            try:
                foreign_keys = inspector.get_foreign_keys(table_name)
                relations_found += len(foreign_keys)
            except:
                continue

        test_results["relations_found"] = relations_found
        test_results["dataframe_compatible"] = True
        test_results["estimated_memory_mb"] = round(test_results["estimated_total_rows"] * len(table_names) * 0.001, 2)

        logger.info(f"[TEST CONNECTION] Успешное тестирование: {len(test_results['tables_found'])} таблиц")
        return test_results

    except Exception as e:
        logger.error(f"[TEST CONNECTION] Ошибка: {e}")
        return {
            "connection_status": "failed",
            "error": str(e),
            "dataframe_compatible": False
        }

    finally:
        try:
            db_session.close()
        except:
            pass


@celery_app.task(bind=True, time_limit=300, name='tasks.cleanup_old_reports')
def cleanup_old_reports(self, days_old: int = 30):
    """Очищает старые отчеты для экономии места в БД"""

    db_session = next(database.get_db())

    try:
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days_old)

        # Находим старые отчеты
        old_reports = db_session.query(crud.models.Report).filter(
            crud.models.Report.created_at < cutoff_date,
            crud.models.Report.status.in_(['COMPLETED', 'FAILED'])
        ).all()

        cleaned_count = 0
        for report in old_reports:
            try:
                # Сохраняем только metadata, удаляем detailed_findings
                if report.results and isinstance(report.results, dict):
                    simplified_results = {
                        "executive_summary": report.results.get("executive_summary", ""),
                        "analysis_stats": report.results.get("analysis_stats", {}),
                        "method": report.results.get("method", ""),
                        "cleaned_date": datetime.now().isoformat(),
                        "original_size": "cleaned"
                    }
                    report.results = simplified_results
                    cleaned_count += 1
            except Exception as clean_error:
                logger.error(f"Ошибка очистки отчета {report.id}: {clean_error}")

        db_session.commit()

        logger.info(f"[CLEANUP] Очищено {cleaned_count} старых отчетов (старше {days_old} дней)")

        return {
            "status": "success",
            "cleaned_reports": cleaned_count,
            "cutoff_date": cutoff_date.isoformat()
        }

    except Exception as e:
        logger.error(f"[CLEANUP] Ошибка очистки: {e}")
        db_session.rollback()
        return {
            "status": "failed",
            "error": str(e)
        }

    finally:
        try:
            db_session.close()
        except:
            pass


# =============== МОНИТОРИНГ И СТАТИСТИКА ===============

@celery_app.task(bind=True, time_limit=180, name='tasks.get_system_health')
def get_system_health(self):
    """Проверяет здоровье системы DataFrame анализа"""

    try:
        # Проверяем Celery
        i = celery_app.control.inspect()
        active_tasks = i.active()
        scheduled_tasks = i.scheduled()

        active_count = sum(len(tasks) for tasks in (active_tasks or {}).values())
        scheduled_count = sum(len(tasks) for tasks in (scheduled_tasks or {}).values())

        # Проверяем базу данных
        db_session = next(database.get_db())

        try:
            # Подсчет отчетов по статусам
            from sqlalchemy import func
            report_stats = db_session.query(
                crud.models.Report.status,
                func.count(crud.models.Report.id)
            ).group_by(crud.models.Report.status).all()

            status_counts = dict(report_stats)

            # Последние отчеты
            recent_reports = db_session.query(crud.models.Report).filter(
                crud.models.Report.created_at >= datetime.now() - timedelta(hours=24)
            ).count()

            health_status = {
                "timestamp": datetime.now().isoformat(),
                "celery": {
                    "active_tasks": active_count,
                    "scheduled_tasks": scheduled_count,
                    "status": "healthy" if active_count < 10 else "busy"
                },
                "database": {
                    "status": "healthy",
                    "recent_reports_24h": recent_reports,
                    "total_completed": status_counts.get("COMPLETED", 0),
                    "total_failed": status_counts.get("FAILED", 0),
                    "total_processing": status_counts.get("PROCESSING", 0)
                },
                "dataframe_system": {
                    "status": "operational",
                    "supported_features": [
                        "automatic_table_loading",
                        "relationship_detection",
                        "anomaly_detection",
                        "correlation_analysis",
                        "trend_analysis",
                        "memory_optimization"
                    ]
                }
            }

            # Вычисляем общий статус
            if status_counts.get("FAILED", 0) > status_counts.get("COMPLETED", 0):
                health_status["overall_status"] = "warning"
            elif active_count > 15:
                health_status["overall_status"] = "busy"
            else:
                health_status["overall_status"] = "healthy"

        finally:
            db_session.close()

        return health_status

    except Exception as e:
        logger.error(f"[HEALTH CHECK] Ошибка: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "error",
            "error": str(e)
        }


# =============== ЭКСПОРТ И ИНТЕГРАЦИЯ ===============

@celery_app.task(bind=True, time_limit=1800, name='tasks.export_report_to_excel')
def export_report_to_excel(self, report_id: int, user_id: int):
    """Экспортирует отчет в Excel файл"""

    db_session = next(database.get_db())

    try:
        # Получаем отчет
        report = crud.get_report_by_id(db_session, report_id, user_id)
        if not report or not report.results:
            raise ValueError("Отчет не найден или пуст")

        # Создаем Excel файл
        import io
        from datetime import datetime

        output = io.BytesIO()

        # Здесь должна быть логика создания Excel файла
        # Пока возвращаем информацию о том, что экспорт готов

        export_info = {
            "status": "completed",
            "report_id": report_id,
            "export_date": datetime.now().isoformat(),
            "file_size_bytes": len(str(report.results)),
            "format": "excel",
            "note": "Excel export feature - в разработке"
        }

        return export_info

    except Exception as e:
        logger.error(f"[EXCEL EXPORT] Ошибка: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "report_id": report_id
        }

    finally:
        try:
            db_session.close()
        except:
            pass


logger.info("DataFrame tasks система полностью загружена")
