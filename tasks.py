# tasks.py - полная DataFrame система с GPT интеграцией
import logging
from typing import List, Dict, Any
import numpy as np
from celery.exceptions import Ignore
from sqlalchemy import create_engine, text
import crud
import database
from services.dataframe_manager import DataFrameManager
from services.dataframe_analyzer import DataFrameAnalyzer
from services.gpt_analyzer import GPTAnalyzer
from utils.json_serializer import convert_to_serializable
from celery_worker import celery_app
from datetime import datetime, timedelta
import json
import sys
import os

# Добавляем путь для импорта utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, time_limit=7200, name='tasks.generate_dataframe_report')
def generate_dataframe_report(self, connection_id: int, user_id: int, report_id: int, max_questions: int = 15):
    """
    Генерация отчета полностью на основе DataFrame с GPT анализом
    """
    db_session = next(database.get_db())

    try:
        logger.info(f"[DATAFRAME REPORT] Запуск для пользователя {user_id}, отчет {report_id}")

        # === ИНИЦИАЛИЗАЦИЯ ===
        self.update_state(
            state='INITIALIZING',
            meta={'progress': 'Инициализация DataFrame системы с GPT...'}
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
        tables_loaded = df_manager.load_all_tables(max_rows_per_table=50000)

        if not tables_loaded:
            raise ValueError("Не удалось загрузить данные из базы. Проверьте подключение и права доступа.")

        logger.info(f"[DATAFRAME REPORT] Успешно загружено {len(tables_loaded)} таблиц")

        # === ОПТИМИЗАЦИЯ ПАМЯТИ ===
        self.update_state(
            state='OPTIMIZING',
            meta={'progress': 'Оптимизация использования памяти...', 'progress_percentage': 20}
        )

        df_manager.optimize_memory()

        # === СОЗДАНИЕ АНАЛИЗАТОРОВ ===
        self.update_state(
            state='INITIALIZING_ANALYZERS',
            meta={'progress': 'Инициализация GPT и DataFrame анализаторов...', 'progress_percentage': 25}
        )

        analyzer = DataFrameAnalyzer(df_manager)
        gpt_analyzer = GPTAnalyzer()

        # === СОЗДАНИЕ ИНТЕЛЛЕКТУАЛЬНОГО ПЛАНА АНАЛИЗА ===
        self.update_state(
            state='PLANNING',
            meta={'progress': 'Создание интеллектуального плана анализа...', 'progress_percentage': 30}
        )

        analysis_plan = _create_enhanced_analysis_plan(df_manager, max_questions)
        logger.info(f"[DATAFRAME REPORT] План анализа: {len(analysis_plan)} вопросов")

        # === ВЫПОЛНЕНИЕ АНАЛИЗА ===
        session_memory = []
        successful_analyses = 0
        gpt_analyses_count = 0

        for i, question_config in enumerate(analysis_plan):
            if i >= max_questions:
                break

            progress = 30 + (i + 1) / min(len(analysis_plan), max_questions) * 60  # 30-90%
            question = question_config.get('question', str(question_config))
            analysis_type = question_config.get('type', 'general')

            self.update_state(
                state='ANALYZING',
                meta={
                    'progress': f'Анализ {i + 1}/{min(len(analysis_plan), max_questions)}: {question}',
                    'progress_percentage': progress,
                    'analysis_type': analysis_type
                }
            )

            logger.info(f"[DATAFRAME REPORT] Анализ {i + 1}: {question} (тип: {analysis_type})")

            try:
                # Выполняем DataFrame анализ
                result = analyzer.analyze_question(question)

                if not result.get('error'):
                    # Безопасная подготовка данных для сериализации
                    data_preview = result.get('data', None)
                    if data_preview:
                        if hasattr(data_preview, 'head'):
                            data_preview = convert_to_serializable(data_preview.head(10).to_dict('records'))
                        elif isinstance(data_preview, (list, dict)):
                            data_preview = convert_to_serializable(data_preview)
                    else:
                        data_preview = []

                    # Конвертируем все дополнительные данные
                    additional_info = result.get('additional_info', {})
                    if additional_info:
                        additional_info = convert_to_serializable(additional_info)

                    # === GPT УГЛУБЛЕННЫЙ АНАЛИЗ ===
                    gpt_insights = {}
                    if question_config.get('enable_gpt', True) and data_preview:
                        try:
                            # Определяем тип GPT анализа
                            gpt_type = _determine_gpt_analysis_type(question, analysis_type)

                            if gpt_type and len(result.get('analyzed_tables', [])) > 0:
                                main_table = result['analyzed_tables'][0]
                                df_for_gpt = df_manager.tables[main_table]

                                gpt_result = gpt_analyzer.analyze_data_with_gpt(
                                    df=df_for_gpt,
                                    table_name=main_table,
                                    analysis_type=gpt_type,
                                    context={
                                        'question': question,
                                        'dataframe_results': result,
                                        'analysis_type': analysis_type
                                    }
                                )

                                gpt_insights = {
                                    'gpt_analysis': gpt_result.get('gpt_analysis', ''),
                                    'gpt_type': gpt_type,
                                    'confidence': gpt_result.get('confidence', 'medium')
                                }

                                gpt_analyses_count += 1
                                logger.info(f"[GPT ANALYSIS] Успешно для вопроса {i + 1}")

                        except Exception as gpt_error:
                            logger.error(f"[GPT ANALYSIS] Ошибка для вопроса {i + 1}: {gpt_error}")
                            gpt_insights = {
                                'gpt_analysis': f'GPT анализ недоступен: {str(gpt_error)}',
                                'confidence': 'low'
                            }

                    finding_entry = {
                        'question': str(question),
                        'summary': str(result.get('summary', '')),
                        'data_preview': data_preview,
                        'chart_data': convert_to_serializable(result.get('chart_data')),
                        'analyzed_tables': list(result.get('analyzed_tables', [])),
                        'method': 'dataframe_with_gpt',
                        'analysis_type': analysis_type,
                        'additional_info': additional_info,
                        'gpt_insights': gpt_insights,
                        'timestamp': datetime.now().isoformat(),
                        'success': True
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
                        'analysis_type': analysis_type,
                        'success': False,
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
                    'analysis_type': analysis_type,
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                })

        # === СОЗДАНИЕ ФИНАЛЬНОГО ОТЧЕТА ===
        self.update_state(
            state='FINALIZING',
            meta={'progress': 'Создание финального отчета с GPT сводкой...', 'progress_percentage': 90}
        )

        logger.info(
            f"[DATAFRAME REPORT] Создание итогового отчета: {successful_analyses}/{len(session_memory)} успешных анализов, {gpt_analyses_count} GPT анализов")

        # Получаем сводку по таблицам
        table_summary = df_manager.get_table_summary()
        memory_info = df_manager.get_memory_usage()

        # === GPT EXECUTIVE SUMMARY ===
        executive_summary = ""
        try:
            if successful_analyses > 0:
                gpt_insights = [f for f in session_memory if f.get('gpt_insights', {}).get('gpt_analysis')]
                if gpt_insights:
                    executive_summary = gpt_analyzer.generate_executive_summary(gpt_insights, table_summary)
                else:
                    executive_summary = _create_executive_summary(session_memory, table_summary, successful_analyses)
            else:
                executive_summary = "Анализ не дал результатов"
        except Exception as summary_error:
            logger.error(f"Ошибка создания executive summary: {summary_error}")
            executive_summary = _create_executive_summary(session_memory, table_summary, successful_analyses)

        # Создаем финальный отчет с безопасной сериализацией
        final_report = {
            "executive_summary": str(executive_summary),
            "detailed_findings": convert_to_serializable(session_memory),
            "method": "dataframe_with_gpt",
            "tables_info": convert_to_serializable(table_summary['tables']),
            "relations_info": convert_to_serializable(table_summary['relations']),
            "analysis_stats": {
                "questions_processed": int(len(session_memory)),
                "successful_analyses": int(successful_analyses),
                "failed_analyses": int(len(session_memory) - successful_analyses),
                "gpt_analyses_count": int(gpt_analyses_count),
                "tables_analyzed": int(table_summary['total_tables']),
                "relations_found": int(table_summary['total_relations']),
                "total_memory_mb": round(float(table_summary['total_memory_mb']), 2),
                "success_rate_percent": round(float(successful_analyses / max(len(session_memory), 1) * 100), 1),
                "gpt_integration": True
            },
            "memory_usage": convert_to_serializable(memory_info),
            "recommendations": [str(r) for r in
                                _generate_enhanced_recommendations(session_memory, table_summary, successful_analyses,
                                                                   gpt_analyses_count)],
            "report_metadata": {
                "created_at": datetime.now().isoformat(),
                "user_id": int(user_id),
                "connection_id": int(connection_id),
                "report_version": "3.0_dataframe_gpt",
                "max_questions_requested": int(max_questions),
                "gpt_enabled": True
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

            try:
                db_session.rollback()
            except:
                pass

            # Попытка сохранить упрощенный отчет
            simplified_report = {
                "executive_summary": final_report["executive_summary"],
                "method": "dataframe_with_gpt",
                "analysis_stats": final_report["analysis_stats"],
                "error": f"Полный отчет не удалось сохранить: {str(save_error)}",
                "report_metadata": final_report["report_metadata"]
            }

            crud.update_report(db_session, report_id, "COMPLETED", simplified_report)
            logger.warning(f"[DATAFRAME REPORT] Сохранен упрощенный отчет {report_id}")

        self.update_state(
            state='SUCCESS',
            meta={'progress': 'DataFrame-анализ с GPT завершен успешно!', 'progress_percentage': 100}
        )

        return {
            "status": "success",
            "report_id": report_id,
            "questions_processed": len(session_memory),
            "successful_analyses": successful_analyses,
            "gpt_analyses": gpt_analyses_count,
            "tables_loaded": len(tables_loaded)
        }

    except Exception as e:
        logger.error(f"[DATAFRAME REPORT ERROR] Критическая ошибка: {e}", exc_info=True)

        try:
            db_session.rollback()
        except:
            pass

        error_report = {
            "error": str(e),
            "method": "dataframe_with_gpt",
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


def _create_enhanced_analysis_plan(df_manager: DataFrameManager, max_questions: int) -> List[Dict[str, Any]]:
    """Создает интеллектуальный план анализа с GPT интеграцией"""

    plan = [
        {
            'question': 'Общий обзор структуры данных и связей между таблицами',
            'type': 'overview',
            'enable_gpt': True,
            'priority': 1
        }
    ]

    table_names = list(df_manager.tables.keys())

    # Анализ каждой важной таблицы
    tables_by_size = sorted(df_manager.tables.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (table_name, df) in enumerate(tables_by_size[:min(4, len(tables_by_size))]):
        if i < 2:  # Первые 2 таблицы - детальный GPT анализ
            plan.extend([
                {
                    'question': f"Детальный бизнес-анализ таблицы '{table_name}' с GPT инсайтами",
                    'type': 'table_analysis',
                    'table_focus': table_name,
                    'enable_gpt': True,
                    'priority': 2
                },
                {
                    'question': f"Анализ качества данных таблицы '{table_name}' с рекомендациями",
                    'type': 'data_quality',
                    'table_focus': table_name,
                    'enable_gpt': True,
                    'priority': 2
                }
            ])
        else:  # Остальные - базовый анализ
            plan.append({
                'question': f"Экспресс-анализ структуры таблицы '{table_name}'",
                'type': 'table_analysis',
                'table_focus': table_name,
                'enable_gpt': False,
                'priority': 3
            })

    # Анализ связей если они есть
    if df_manager.relations:
        plan.append({
            'question': 'Анализ качества связей между таблицами и их целостности',
            'type': 'relationship_analysis',
            'enable_gpt': True,
            'priority': 2
        })

    # Специфические анализы в зависимости от данных
    has_numeric_data = any(
        len(df.select_dtypes(include=[np.number]).columns) > 0
        for df in df_manager.tables.values()
    )

    has_datetime_data = any(
        any('date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() for col in df.columns)
        for df in df_manager.tables.values()
    )

    if has_numeric_data:
        plan.extend([
            {
                'question': 'Глубокий статистический анализ числовых данных с GPT интерпретацией',
                'type': 'statistical_insights',
                'enable_gpt': True,
                'priority': 2
            },
            {
                'question': 'Анализ корреляций между переменными с бизнес-выводами',
                'type': 'correlation',
                'enable_gpt': True,
                'priority': 2
            },
            {
                'question': 'Поиск аномалий и выбросов с анализом причин',
                'type': 'anomalies',
                'enable_gpt': True,
                'priority': 3
            }
        ])

    if has_datetime_data:
        plan.append({
            'question': 'Предиктивный анализ временных трендов и прогнозирование',
            'type': 'predictive_analysis',
            'enable_gpt': True,
            'priority': 2
        })

    # Общие GPT-анализы
    plan.extend([
        {
            'question': 'Анализ бизнес-метрик и KPI с рекомендациями',
            'type': 'business_insights',
            'enable_gpt': True,
            'priority': 2
        },
        {
            'question': 'Сравнительный анализ таблиц с выявлением паттернов',
            'type': 'comparison',
            'enable_gpt': True,
            'priority': 3
        }
    ])

    # Дополнительные анализы для больших планов
    if max_questions > 15:
        plan.extend([
            {
                'question': 'Поиск скрытых бизнес-возможностей в данных',
                'type': 'business_insights',
                'enable_gpt': True,
                'priority': 3
            },
            {
                'question': 'Анализ потенциала для машинного обучения',
                'type': 'predictive_analysis',
                'enable_gpt': True,
                'priority': 4
            },
            {
                'question': 'Рекомендации по оптимизации структуры данных',
                'type': 'data_quality',
                'enable_gpt': True,
                'priority': 4
            }
        ])

    # Сортируем по приоритету и возвращаем нужное количество
    plan.sort(key=lambda x: x.get('priority', 5))
    return plan[:max_questions]


def _determine_gpt_analysis_type(question: str, analysis_type: str) -> str:
    """Определяет тип GPT анализа на основе вопроса"""

    question_lower = question.lower()

    if analysis_type == 'overview':
        return 'business_insights'
    elif analysis_type == 'table_analysis':
        if 'бизнес' in question_lower or 'инсайт' in question_lower:
            return 'business_insights'
        elif 'качество' in question_lower or 'проблем' in question_lower:
            return 'data_quality'
        else:
            return 'business_insights'
    elif analysis_type == 'statistical_insights':
        return 'statistical_insights'
    elif analysis_type == 'predictive_analysis':
        return 'predictive_analysis'
    elif analysis_type == 'data_quality':
        return 'data_quality'
    elif analysis_type == 'business_insights':
        return 'business_insights'
    elif analysis_type == 'correlation':
        return 'statistical_insights'
    elif analysis_type == 'anomalies':
        return 'data_quality'
    else:
        return 'business_insights'


def _generate_enhanced_recommendations(session_memory: List[dict], table_summary: dict,
                                       successful_analyses: int, gpt_analyses_count: int) -> List[str]:
    """Генерирует улучшенные рекомендации с учетом GPT анализа"""

    recommendations = []

    try:
        total_tables = table_summary.get('total_tables', 0)
        total_relations = table_summary.get('total_relations', 0)
        total_memory = table_summary.get('total_memory_mb', 0)

        # Рекомендации по результатам GPT анализа
        if gpt_analyses_count > 0:
            recommendations.append(f"🤖 Проведено {gpt_analyses_count} углубленных GPT-анализов с детальными инсайтами")

            # Собираем ключевые insights из GPT анализов
            gpt_insights = []
            for finding in session_memory:
                gpt_data = finding.get('gpt_insights', {})
                if gpt_data.get('gpt_analysis'):
                    gpt_insights.append(gpt_data['gpt_analysis'])

            if len(gpt_insights) >= 3:
                recommendations.append(
                    "📊 GPT анализ выявил значительные бизнес-возможности - детали в разделе инсайтов")

        # Стандартные рекомендации
        success_rate = (successful_analyses / max(len(session_memory), 1)) * 100

        if success_rate > 90:
            recommendations.append("✅ Отличная успешность анализа - данные высокого качества для принятия решений")
        elif success_rate < 70:
            recommendations.append("⚠️ Низкая успешность анализа - рекомендуется проверить качество и структуру данных")

        # Рекомендации по структуре данных
        if total_relations > 0:
            recommendations.append(
                f"🔗 Обнаружено {total_relations} связей между таблицами - используйте для создания интегрированных дашбордов")
        elif total_tables > 1:
            recommendations.append(
                "❗ Связи между таблицами не обнаружены - настройте внешние ключи для обеспечения целостности")

        # Рекомендации по производительности
        if total_memory > 500:
            recommendations.append(
                "🚀 Большой объем данных в памяти - рассмотрите использование индексов и кэширования")

        # Рекомендации по автоматизации
        recommendations.extend([
            "🔄 Настройте регулярное обновление DataFrame-анализа для мониторинга изменений",
            "📈 Создайте дашборды на основе выявленных ключевых метрик",
            "🤖 Используйте GPT-инсайты для создания персонализированных отчетов",
            "🔔 Настройте систему алертов на основе обнаруженных аномалий"
        ])

        return recommendations[:10]

    except Exception as e:
        logger.error(f"Ошибка генерации рекомендаций: {e}")
        return [
            "Завершен расширенный DataFrame-анализ с GPT интеграцией",
            "Рекомендуется регулярное обновление для получения актуальных инсайтов"
        ]


# Остальные вспомогательные функции остаются такими же как в исходном коде...
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
            f"Проведен полный DataFrame-анализ с GPT интеграцией.",
            f"Обработано {successful_analyses} из {total_questions} аналитических вопросов (успешность: {success_rate:.1f}%).",
            f"Загружено {total_tables} таблиц ({total_memory:.1f} MB) с {total_relations} связями.",
            f"Проанализировано {len(analyzed_tables)} уникальных таблиц."
        ]

        return " ".join(summary_parts)

    except Exception as e:
        logger.error(f"Ошибка создания executive summary: {e}")
        return f"DataFrame-анализ завершен с {successful_analyses} успешными анализами из {len(session_memory)} запланированных."


def _trim_large_report(report: dict) -> dict:
    """Сокращает размер слишком большого отчета"""
    try:
        logger.info("Сокращение размера отчета для экономии памяти")

        trimmed_report = report.copy()

        # Сокращаем detailed_findings
        detailed_findings = trimmed_report.get('detailed_findings', [])
        if len(detailed_findings) > 0:
            for finding in detailed_findings:
                # Ограничиваем размер data_preview
                if isinstance(finding.get('data_preview'), list) and len(finding['data_preview']) > 5:
                    finding['data_preview'] = finding['data_preview'][:5]
                    finding['data_preview'].append({"note": f"... показано 5 из множества записей"})

                # Сокращаем GPT анализ если слишком длинный
                gpt_insights = finding.get('gpt_insights', {})
                if gpt_insights.get('gpt_analysis') and len(gpt_insights['gpt_analysis']) > 1000:
                    gpt_insights['gpt_analysis'] = gpt_insights['gpt_analysis'][:1000] + "... (сокращено)"

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
    """Быстрый DataFrame анализ с ограниченным GPT"""
    logger.info(f"[QUICK DATAFRAME] Запуск быстрого анализа для пользователя {user_id}")
    return generate_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 8}
    ).get()


@celery_app.task(bind=True, time_limit=9000, name='tasks.comprehensive_dataframe_analysis')
def comprehensive_dataframe_analysis(self, connection_id: int, user_id: int, report_id: int):
    """Комплексный DataFrame анализ с полным GPT"""
    logger.info(f"[COMPREHENSIVE DATAFRAME] Запуск полного анализа для пользователя {user_id}")
    return generate_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 25}
    ).get()


# =============== LEGACY ПОДДЕРЖКА ===============

@celery_app.task(bind=True, time_limit=3600, name='tasks.generate_advanced_report')
def generate_advanced_report(self, connection_id: int, user_id: int, report_id: int):
    """Legacy функция - перенаправляет на DataFrame анализ"""
    logger.warning(f"[LEGACY] Перенаправление на DataFrame анализ для пользователя {user_id}")
    return generate_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 12}
    ).get()


logger.info("DataFrame tasks система с GPT интеграцией полностью загружена")
