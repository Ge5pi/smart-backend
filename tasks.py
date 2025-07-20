# tasks.py - интеграция со SmartGPTAnalyzer

import logging
from typing import List, Dict, Any
import numpy as np
from celery.exceptions import Ignore
from sqlalchemy import create_engine, text
import crud
import database
from services.dataframe_manager import DataFrameManager
from services.dataframe_analyzer import DataFrameAnalyzer
from services.smart_gpt_analyzer import SmartGPTAnalyzer  # Обновленный импорт
from utils.json_serializer import convert_to_serializable
from celery_worker import celery_app
from datetime import datetime, timedelta
import json
import sys
import os

# Добавляем путь для импорта utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, time_limit=7200, name='tasks.generate_smart_dataframe_report')
def generate_smart_dataframe_report(self, connection_id: int, user_id: int, report_id: int, max_questions: int = 15):
    """
    Генерация отчета с SmartGPT интеграцией - новая архитектура
    """
    db_session = next(database.get_db())

    try:
        logger.info(f"[SMART DATAFRAME] Запуск для пользователя {user_id}, отчет {report_id}")

        # === ИНИЦИАЛИЗАЦИЯ ===
        self.update_state(
            state='INITIALIZING',
            meta={'progress': 'Инициализация SmartGPT DataFrame системы...'}
        )

        connection_string = crud.get_decrypted_connection_string(db_session, connection_id, user_id)
        if not connection_string:
            raise ValueError("Подключение не найдено или недоступно")

        # Создаем подключение только для загрузки данных
        engine = create_engine(connection_string, connect_args={'connect_timeout': 60})

        # === ЗАГРУЗКА ДАННЫХ ===
        self.update_state(
            state='LOADING_DATA',
            meta={'progress': 'Загрузка таблиц с оптимизацией...', 'progress_percentage': 10}
        )

        df_manager = DataFrameManager(engine)
        tables_loaded = df_manager.load_all_tables(max_rows_per_table=50000)

        if not tables_loaded:
            raise ValueError("Не удалось загрузить данные из базы")

        logger.info(f"[SMART DATAFRAME] Загружено {len(tables_loaded)} таблиц")

        # === ОПТИМИЗАЦИЯ И СОЗДАНИЕ АНАЛИЗАТОРОВ ===
        self.update_state(
            state='OPTIMIZING',
            meta={'progress': 'Оптимизация памяти и инициализация SmartGPT...', 'progress_percentage': 20}
        )

        df_manager.optimize_memory()
        analyzer = DataFrameAnalyzer(df_manager)
        smart_gpt = SmartGPTAnalyzer()

        # === СОЗДАНИЕ УМНОГО ПЛАНА АНАЛИЗА ===
        self.update_state(
            state='PLANNING',
            meta={'progress': 'Создание интеллектуального плана с GPT приоритизацией...', 'progress_percentage': 25}
        )

        analysis_plan = _create_smart_analysis_plan(df_manager, max_questions)
        logger.info(f"[SMART DATAFRAME] План: {len(analysis_plan)} умных вопросов")

        # === ВЫПОЛНЕНИЕ УМНОГО АНАЛИЗА ===
        smart_findings = []
        successful_analyses = 0
        gpt_insights_count = 0

        for i, question_config in enumerate(analysis_plan):
            if i >= max_questions:
                break

            progress = 25 + (i + 1) / min(len(analysis_plan), max_questions) * 60  # 25-85%
            question = question_config.get('question', str(question_config))
            analysis_type = question_config.get('type', 'general')

            self.update_state(
                state='SMART_ANALYZING',
                meta={
                    'progress': f'SmartGPT анализ {i + 1}/{min(len(analysis_plan), max_questions)}: {question[:50]}...',
                    'progress_percentage': progress,
                    'analysis_type': analysis_type
                }
            )

            logger.info(f"[SMART DATAFRAME] Анализ {i + 1}: {question} (тип: {analysis_type})")

            try:
                # Выполняем DataFrame анализ с встроенным SmartGPT
                result = analyzer.analyze_question(question)

                if not result.get('error'):
                    # Безопасная сериализация
                    data_preview = result.get('data', [])
                    if hasattr(data_preview, 'head'):
                        data_preview = convert_to_serializable(data_preview.head(10).to_dict('records'))
                    elif isinstance(data_preview, (list, dict)):
                        data_preview = convert_to_serializable(data_preview)
                    else:
                        data_preview = []

                    # Извлекаем SmartGPT инсайты
                    smart_insights = result.get('smart_gpt_insights', {})
                    has_smart_insights = bool(smart_insights.get('business_insights'))

                    if has_smart_insights:
                        gpt_insights_count += 1

                    # Создаем обогащенную запись
                    finding_entry = {
                        'question': str(question),
                        'summary': str(result.get('summary', '')),
                        'data_preview': data_preview[:10],  # Ограничиваем размер
                        'chart_data': convert_to_serializable(result.get('chart_data')),
                        'analyzed_tables': list(result.get('analyzed_tables', [])),
                        'method': 'smart_dataframe_gpt',
                        'analysis_type': analysis_type,

                        # SmartGPT данные
                        'business_insights': smart_insights.get('business_insights', ''),
                        'action_items': smart_insights.get('action_items', []),
                        'risk_assessment': smart_insights.get('risk_assessment', ''),
                        'opportunities': smart_insights.get('opportunities', []),
                        'gpt_confidence': smart_insights.get('confidence', 'medium'),
                        'business_context': smart_insights.get('business_context', {}),

                        # Дополнительные данные
                        'statistical_insights': convert_to_serializable(result.get('statistical_insights', [])),
                        'correlations': convert_to_serializable(result.get('correlations', [])),
                        'quality_metrics': convert_to_serializable(result.get('quality_metrics', [])),
                        'predictive_patterns': convert_to_serializable(result.get('predictive_patterns', [])),

                        'timestamp': datetime.now().isoformat(),
                        'success': True,
                        'has_smart_insights': has_smart_insights
                    }

                    smart_findings.append(finding_entry)
                    successful_analyses += 1

                    logger.info(f"[SMART DATAFRAME] ✅ Умный анализ {i + 1} завершен успешно")

                else:
                    error_msg = str(result.get('error', 'Неизвестная ошибка'))
                    logger.error(f"[SMART DATAFRAME] ❌ Ошибка анализа {i + 1}: {error_msg}")

                    smart_findings.append({
                        'question': str(question),
                        'summary': f'Ошибка анализа: {error_msg}',
                        'data_preview': [],
                        'error': error_msg,
                        'method': 'smart_dataframe_gpt',
                        'analysis_type': analysis_type,
                        'success': False,
                        'timestamp': datetime.now().isoformat(),
                        'has_smart_insights': False
                    })

            except Exception as analysis_error:
                error_msg = f"Критическая ошибка умного анализа: {str(analysis_error)}"
                logger.error(f"[SMART DATAFRAME] 💥 {error_msg}")

                smart_findings.append({
                    'question': str(question),
                    'summary': error_msg,
                    'data_preview': [],
                    'error': error_msg,
                    'method': 'smart_dataframe_gpt',
                    'analysis_type': analysis_type,
                    'success': False,
                    'timestamp': datetime.now().isoformat(),
                    'has_smart_insights': False
                })

        # === СОЗДАНИЕ УМНОГО EXECUTIVE SUMMARY ===
        self.update_state(
            state='GENERATING_SUMMARY',
            meta={'progress': 'Создание умного executive summary...', 'progress_percentage': 85}
        )

        logger.info(
            f"[SMART DATAFRAME] Генерация итогового отчета: {successful_analyses}/{len(smart_findings)} успешных анализов, {gpt_insights_count} SmartGPT инсайтов")

        # Получаем сводку по таблицам
        table_summary = df_manager.get_table_summary()
        memory_info = df_manager.get_memory_usage()

        # === УМНЫЙ EXECUTIVE SUMMARY ===
        executive_summary = ""
        try:
            if successful_analyses > 0 and gpt_insights_count > 0:
                # Собираем все бизнес-инсайты для метасводки
                business_insights = []
                all_action_items = []
                all_risks = []
                all_opportunities = []

                for finding in smart_findings:
                    if finding.get('has_smart_insights'):
                        if finding.get('business_insights'):
                            business_insights.append(finding['business_insights'])
                        if finding.get('action_items'):
                            all_action_items.extend(finding['action_items'])
                        if finding.get('risk_assessment'):
                            all_risks.append(finding['risk_assessment'])
                        if finding.get('opportunities'):
                            all_opportunities.extend(finding['opportunities'])

                # Генерируем метасводку через SmartGPT
                executive_summary = smart_gpt.generate_executive_summary_smart(
                    smart_findings, table_summary
                )

            else:
                executive_summary = _create_fallback_executive_summary(
                    smart_findings, table_summary, successful_analyses
                )

        except Exception as summary_error:
            logger.error(f"Ошибка создания умной сводки: {summary_error}")
            executive_summary = _create_fallback_executive_summary(
                smart_findings, table_summary, successful_analyses
            )

        # === СОЗДАНИЕ ФИНАЛЬНОГО УМНОГО ОТЧЕТА ===
        self.update_state(
            state='FINALIZING',
            meta={'progress': 'Финализация умного отчета...', 'progress_percentage': 90}
        )

        # Создаем финальный отчет с SmartGPT данными
        final_report = {
            "executive_summary": str(executive_summary),
            "detailed_findings": convert_to_serializable(smart_findings),
            "method": "smart_dataframe_gpt_v2",
            "tables_info": convert_to_serializable(table_summary['tables']),
            "relations_info": convert_to_serializable(table_summary['relations']),

            "smart_analysis_stats": {
                "questions_processed": int(len(smart_findings)),
                "successful_analyses": int(successful_analyses),
                "failed_analyses": int(len(smart_findings) - successful_analyses),
                "smart_gpt_insights_count": int(gpt_insights_count),
                "tables_analyzed": int(table_summary['total_tables']),
                "relations_found": int(table_summary['total_relations']),
                "total_memory_mb": round(float(table_summary['total_memory_mb']), 2),
                "success_rate_percent": round(float(successful_analyses / max(len(smart_findings), 1) * 100), 1),
                "smart_gpt_coverage_percent": round(float(gpt_insights_count / max(successful_analyses, 1) * 100), 1)
            },

            "memory_usage": convert_to_serializable(memory_info),
            "smart_recommendations": [str(r) for r in _generate_smart_recommendations(
                smart_findings, table_summary, successful_analyses, gpt_insights_count
            )],

            "report_metadata": {
                "created_at": datetime.now().isoformat(),
                "user_id": int(user_id),
                "connection_id": int(connection_id),
                "report_version": "4.0_smart_dataframe_gpt",
                "max_questions_requested": int(max_questions),
                "smart_gpt_enabled": True,
                "analysis_engine": "SmartGPTAnalyzer"
            }
        }

        # Проверяем размер отчета
        try:
            report_json = json.dumps(final_report, ensure_ascii=False)
            report_size_mb = len(report_json.encode('utf-8')) / 1024 / 1024
            logger.info(f"[SMART DATAFRAME] Размер умного отчета: {report_size_mb:.2f} MB")

            if report_size_mb > 15:  # Увеличенный лимит для умных отчетов
                logger.warning(f"[SMART DATAFRAME] Отчет слишком большой ({report_size_mb:.2f} MB), сокращаем")
                final_report = _trim_smart_report(final_report)

        except Exception as json_error:
            logger.error(f"[SMART DATAFRAME] Ошибка JSON сериализации: {json_error}")
            final_report = convert_to_serializable(final_report)

        # === СОХРАНЕНИЕ УМНОГО ОТЧЕТА ===
        self.update_state(
            state='SAVING',
            meta={'progress': 'Сохранение умного отчета...', 'progress_percentage': 95}
        )

        try:
            crud.update_report(db_session, report_id, "COMPLETED", final_report)
            logger.info(f"[SMART DATAFRAME] ✅ Умный отчет {report_id} успешно сохранен")

        except Exception as save_error:
            logger.error(f"[SMART DATAFRAME] Ошибка сохранения: {save_error}")
            # Сохраняем упрощенный отчет в случае ошибки
            simplified_report = {
                "executive_summary": final_report["executive_summary"],
                "method": "smart_dataframe_gpt_v2",
                "smart_analysis_stats": final_report["smart_analysis_stats"],
                "error": f"Полный отчет не сохранен: {str(save_error)}",
                "report_metadata": final_report["report_metadata"]
            }
            crud.update_report(db_session, report_id, "COMPLETED", simplified_report)

        self.update_state(
            state='SUCCESS',
            meta={'progress': 'SmartGPT DataFrame-анализ завершен!', 'progress_percentage': 100}
        )

        return {
            "status": "success",
            "report_id": report_id,
            "questions_processed": len(smart_findings),
            "successful_analyses": successful_analyses,
            "smart_gpt_insights": gpt_insights_count,
            "tables_loaded": len(tables_loaded),
            "method": "smart_dataframe_gpt_v2"
        }

    except Exception as e:
        logger.error(f"[SMART DATAFRAME ERROR] Критическая ошибка: {e}", exc_info=True)

        # Сохраняем информацию об ошибке
        error_report = {
            "error": str(e),
            "method": "smart_dataframe_gpt_v2",
            "stage": "critical_error",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "connection_id": connection_id
        }

        try:
            crud.update_report(db_session, report_id, "FAILED", error_report)
        except Exception as save_error:
            logger.error(f"[SMART DATAFRAME] Не удалось сохранить ошибку: {save_error}")

        self.update_state(
            state='FAILURE',
            meta={
                'progress': f'Критическая ошибка SmartGPT: {str(e)}',
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


def _create_smart_analysis_plan(df_manager: DataFrameManager, max_questions: int) -> List[Dict[str, Any]]:
    """Создает умный план анализа для SmartGPT"""

    plan = [
        {
            'question': 'Умный обзор структуры данных с бизнес-контекстом и возможностями',
            'type': 'overview',
            'enable_smart_gpt': True,
            'priority': 1
        }
    ]

    # Анализ каждой важной таблицы с SmartGPT инсайтами
    tables_by_size = sorted(df_manager.tables.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (table_name, df) in enumerate(tables_by_size[:min(3, len(tables_by_size))]):
        plan.extend([
            {
                'question': f"Умный бизнес-анализ таблицы '{table_name}' с выявлением скрытых возможностей",
                'type': 'business_insights',
                'table_focus': table_name,
                'enable_smart_gpt': True,
                'priority': 1 if i == 0 else 2
            },
            {
                'question': f"Анализ качества данных таблицы '{table_name}' с умными рекомендациями",
                'type': 'data_quality',
                'table_focus': table_name,
                'enable_smart_gpt': True,
                'priority': 2
            }
        ])

    # Специализированные SmartGPT анализы
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
            {
                'question': 'Углубленный статистический анализ с SmartGPT интерпретацией паттернов',
                'type': 'statistical_insights',
                'enable_smart_gpt': True,
                'priority': 2
            },
            {
                'question': 'Анализ корреляций с бизнес-интерпретацией и выводами',
                'type': 'correlation',
                'enable_smart_gpt': True,
                'priority': 2
            }
        ])

    if has_datetime_data:
        plan.extend([
            {
                'question': 'Умный анализ временных трендов с предсказаниями и возможностями',
                'type': 'predictive_analysis',
                'enable_smart_gpt': True,
                'priority': 2
            }
        ])

    # Связи между таблицами
    if df_manager.relations:
        plan.append({
            'question': 'Анализ целостности связей с умными рекомендациями по оптимизации',
            'type': 'relationship_analysis',
            'enable_smart_gpt': True,
            'priority': 2
        })

    # Поиск аномалий и возможностей
    plan.extend([
        {
            'question': 'Умный поиск аномалий с анализом причин и влияния на бизнес',
            'type': 'anomalies',
            'enable_smart_gpt': True,
            'priority': 3
        },
        {
            'question': 'Сравнительный анализ таблиц с выявлением скрытых паттернов',
            'type': 'comparison',
            'enable_smart_gpt': True,
            'priority': 3
        }
    ])

    # Дополнительные анализы для больших планов
    if max_questions > 15:
        plan.extend([
            {
                'question': 'Поиск скрытых бизнес-возможностей и точек роста через данные',
                'type': 'business_insights',
                'enable_smart_gpt': True,
                'priority': 3
            },
            {
                'question': 'Оценка потенциала для машинного обучения и автоматизации',
                'type': 'predictive_analysis',
                'enable_smart_gpt': True,
                'priority': 4
            }
        ])

    # Сортируем по приоритету и возвращаем нужное количество
    plan.sort(key=lambda x: x.get('priority', 5))
    return plan[:max_questions]


def _generate_smart_recommendations(smart_findings: List[dict], table_summary: dict,
                                    successful_analyses: int, gpt_insights_count: int) -> List[str]:
    """Генерирует умные рекомендации на основе SmartGPT инсайтов"""

    recommendations = []

    # Анализируем SmartGPT инсайты
    all_action_items = []
    all_opportunities = []
    high_confidence_insights = 0

    for finding in smart_findings:
        if finding.get('has_smart_insights'):
            if finding.get('action_items'):
                all_action_items.extend(finding['action_items'])
            if finding.get('opportunities'):
                all_opportunities.extend(finding['opportunities'])
            if finding.get('gpt_confidence') == 'high':
                high_confidence_insights += 1

    # Основные рекомендации на основе SmartGPT
    if gpt_insights_count > 0:
        recommendations.append(
            f"🤖 Получено {gpt_insights_count} умных GPT-инсайтов с практическими рекомендациями"
        )

        if high_confidence_insights > gpt_insights_count * 0.7:
            recommendations.append(
                "✅ Высокое качество данных - GPT-инсайты имеют высокую достоверность"
            )

    # Приоритетные действия
    if all_action_items:
        top_actions = list(set(all_action_items))[:3]  # Уникальные топ-3
        recommendations.append(
            f"🎯 Приоритетные действия выявлены: {len(top_actions)} конкретных рекомендаций"
        )

    # Возможности роста
    if all_opportunities:
        unique_opportunities = list(set(all_opportunities))[:3]
        recommendations.append(
            f"🚀 Обнаружено {len(unique_opportunities)} возможностей для роста бизнеса"
        )

    # Рекомендации по автоматизации
    success_rate = (successful_analyses / max(len(smart_findings), 1)) * 100
    gpt_coverage = (gpt_insights_count / max(successful_analyses, 1)) * 100

    if success_rate > 85 and gpt_coverage > 70:
        recommendations.extend([
            "🔄 Настройте регулярные SmartGPT отчеты для автоматического мониторинга",
            "📊 Создайте дашборды на основе выявленных ключевых метрик",
            "🔔 Настройте алерты на основе найденных аномалий и трендов"
        ])

    # Рекомендации по данным
    total_memory = table_summary.get('total_memory_mb', 0)
    if total_memory > 1000:
        recommendations.append(
            "⚡ Рассмотрите оптимизацию хранения данных - большой объем в памяти"
        )

    # Общие рекомендации
    recommendations.extend([
        "🎨 Визуализируйте ключевые инсайты для презентации стейкхолдерам",
        "📈 Используйте выявленные паттерны для стратегического планирования",
        "🤝 Поделитесь бизнес-инсайтами с соответствующими командами"
    ])

    return recommendations[:8]  # Ограничиваем количество


def _create_fallback_executive_summary(smart_findings: List[dict], table_summary: dict,
                                       successful_analyses: int) -> str:
    """Создает резервное executive summary если SmartGPT недоступен"""

    total_questions = len(smart_findings)
    total_tables = table_summary.get('total_tables', 0)
    total_relations = table_summary.get('total_relations', 0)

    return (
        f"Завершен умный DataFrame-анализ с {successful_analyses} успешными анализами "
        f"из {total_questions} запланированных. Проанализировано {total_tables} таблиц "
        f"с {total_relations} связями. SmartGPT инсайты формируются..."
    )


def _trim_smart_report(report: dict) -> dict:
    """Сокращает размер умного отчета"""
    try:
        trimmed = report.copy()

        # Сокращаем detailed_findings
        if 'detailed_findings' in trimmed and isinstance(trimmed['detailed_findings'], list):
            for finding in trimmed['detailed_findings']:
                # Ограничиваем размер data_preview
                if 'data_preview' in finding and isinstance(finding['data_preview'], list):
                    if len(finding['data_preview']) > 3:
                        finding['data_preview'] = finding['data_preview'][:3]
                        finding['data_preview'].append({"note": "... данные сокращены"})

                # Сокращаем длинные текстовые инсайты
                for text_field in ['business_insights', 'risk_assessment']:
                    if finding.get(text_field) and len(finding[text_field]) > 800:
                        finding[text_field] = finding[text_field][:800] + "... (сокращено)"

        trimmed['report_metadata']['trimmed'] = True
        trimmed['report_metadata']['trim_reason'] = "Отчет сокращен для оптимизации размера"

        return trimmed

    except Exception as e:
        logger.error(f"Ошибка сокращения умного отчета: {e}")
        return report


# Остальные задачи - быстрый и комплексный анализ
@celery_app.task(bind=True, time_limit=3600, name='tasks.quick_smart_analysis')
def quick_smart_analysis(self, connection_id: int, user_id: int, report_id: int):
    """Быстрый SmartGPT анализ"""
    logger.info(f"[QUICK SMART] Запуск быстрого SmartGPT анализа для пользователя {user_id}")
    return generate_smart_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 8}
    ).get()


@celery_app.task(bind=True, time_limit=9000, name='tasks.comprehensive_smart_analysis')
def comprehensive_smart_analysis(self, connection_id: int, user_id: int, report_id: int):
    """Комплексный SmartGPT анализ"""
    logger.info(f"[COMPREHENSIVE SMART] Запуск полного SmartGPT анализа для пользователя {user_id}")
    return generate_smart_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 25}
    ).get()


# Legacy совместимость
@celery_app.task(bind=True, time_limit=7200, name='tasks.generate_dataframe_report')
def generate_dataframe_report(self, connection_id: int, user_id: int, report_id: int, max_questions: int = 15):
    """Legacy функция - перенаправляет на SmartGPT анализ"""
    logger.warning(f"[LEGACY] Перенаправление на SmartGPT анализ для пользователя {user_id}")
    return generate_smart_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': max_questions}
    ).get()


logger.info("SmartGPT DataFrame tasks система полностью загружена")
