# tasks.py - SmartGPT DataFrame система

import logging
from typing import List, Dict, Any
import numpy as np
from celery.exceptions import Ignore
from sqlalchemy import create_engine, text
import crud
import database
from services.dataframe_manager import DataFrameManager
from services.dataframe_analyzer import DataFrameAnalyzer
from services.gpt_analyzer import SmartGPTAnalyzer  # Обновленный импорт
from utils.json_serializer import convert_to_serializable
from celery_worker import celery_app
from datetime import datetime, timedelta
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, time_limit=7200, name='tasks.generate_dataframe_report')
def generate_dataframe_report(self, connection_id: int, user_id: int, report_id: int, max_questions: int = 15):
    """Основная задача генерации SmartGPT DataFrame отчета"""

    db_session = next(database.get_db())

    try:
        logger.info(f"[SMARTGPT DATAFRAME] 🚀 Запуск для пользователя {user_id}, отчет {report_id}")

        # === ИНИЦИАЛИЗАЦИЯ ===
        self.update_state(
            state='INITIALIZING',
            meta={'progress': '🔧 Инициализация SmartGPT DataFrame системы...', 'progress_percentage': 5}
        )

        connection_string = crud.get_decrypted_connection_string(db_session, connection_id, user_id)
        if not connection_string:
            raise ValueError("Подключение не найдено или недоступно")

        engine = create_engine(connection_string, connect_args={'connect_timeout': 60})

        # === ЗАГРУЗКА ДАННЫХ ===
        self.update_state(
            state='LOADING_DATA',
            meta={'progress': '📊 Загрузка всех таблиц в память для SmartGPT анализа...', 'progress_percentage': 10}
        )

        df_manager = DataFrameManager(engine)
        tables_loaded = df_manager.load_all_tables(max_rows_per_table=50000)

        if not tables_loaded:
            raise ValueError("Не удалось загрузить данные из базы для SmartGPT анализа")

        logger.info(f"[SMARTGPT DATAFRAME] ✅ Загружено {len(tables_loaded)} таблиц")

        # === ОПТИМИЗАЦИЯ И ИНИЦИАЛИЗАЦИЯ АНАЛИЗАТОРОВ ===
        self.update_state(
            state='OPTIMIZING',
            meta={'progress': '⚡ Оптимизация памяти и инициализация SmartGPT...', 'progress_percentage': 20}
        )

        df_manager.optimize_memory()
        analyzer = DataFrameAnalyzer(df_manager)
        smart_gpt = SmartGPTAnalyzer()

        # === СОЗДАНИЕ ИНТЕЛЛЕКТУАЛЬНОГО ПЛАНА ===
        self.update_state(
            state='PLANNING',
            meta={'progress': '🧠 Создание умного плана анализа с GPT приоритизацией...', 'progress_percentage': 25}
        )

        analysis_plan = _create_smartgpt_analysis_plan(df_manager, max_questions)
        logger.info(f"[SMARTGPT DATAFRAME] 📋 План создан: {len(analysis_plan)} умных вопросов")

        # === ВЫПОЛНЕНИЕ SMARTGPT АНАЛИЗА ===
        smartgpt_findings = []
        successful_analyses = 0
        gpt_insights_count = 0

        for i, question_config in enumerate(analysis_plan):
            if i >= max_questions:
                break

            progress = 25 + (i + 1) / min(len(analysis_plan), max_questions) * 60  # 25-85%
            question = question_config.get('question', str(question_config))
            analysis_type = question_config.get('type', 'general')
            enable_gpt = question_config.get('enable_gpt', True)

            self.update_state(
                state='SMART_ANALYZING',
                meta={
                    'progress': f'🤖 SmartGPT анализ {i + 1}/{min(len(analysis_plan), max_questions)}: {question[:60]}...',
                    'progress_percentage': progress,
                    'analysis_type': analysis_type
                }
            )

            logger.info(f"[SMARTGPT DATAFRAME] 🔍 Анализ {i + 1}: {question} (тип: {analysis_type})")

            try:
                # Выполняем DataFrame анализ
                result = analyzer.analyze_question(question)

                if not result.get('error'):
                    # Безопасная подготовка данных
                    data_preview = result.get('data', [])
                    if hasattr(data_preview, 'head'):
                        data_preview = convert_to_serializable(data_preview.head(10).to_dict('records'))
                    elif isinstance(data_preview, (list, dict)):
                        data_preview = convert_to_serializable(data_preview)
                    else:
                        data_preview = []

                    # === SMARTGPT ОБОГАЩЕНИЕ ===
                    smartgpt_insights = {}
                    if enable_gpt and data_preview and result.get('analyzed_tables'):
                        try:
                            main_table = result['analyzed_tables'][0]
                            df_for_gpt = df_manager.tables[main_table]

                            # Определяем тип SmartGPT анализа
                            gpt_type = _map_analysis_to_smartgpt_type(question, analysis_type)

                            # Создаем бизнес-контекст
                            business_context = {
                                'question': question,
                                'dataframe_results': result,
                                'analysis_type': analysis_type,
                                'table_focus': main_table,
                                'user_intent': _extract_user_intent(question)
                            }

                            # Получаем SmartGPT инсайты
                            gpt_result = smart_gpt.analyze_findings_with_context(df=df_for_gpt,
                                                                                 dataframe_results=result,
                                                                                 business_context=business_context
                                                                                 )

                            smartgpt_insights = {
                                'business_insights': gpt_result.get('business_insights', ''),
                                'action_items': gpt_result.get('action_items', []),
                                'risk_assessment': gpt_result.get('risk_assessment', ''),
                                'opportunities': gpt_result.get('opportunities', []),
                                'confidence': gpt_result.get('confidence', 'medium'),
                                'business_context': gpt_result.get('business_context', {})
                            }

                            gpt_insights_count += 1
                            logger.info(f"[SMARTGPT] ✨ Умные инсайты получены для вопроса {i + 1}")

                        except Exception as gpt_error:
                            logger.error(f"[SMARTGPT] ❌ Ошибка GPT анализа для вопроса {i + 1}: {gpt_error}")
                            smartgpt_insights = {
                                'business_insights': f'SmartGPT анализ временно недоступен: {str(gpt_error)}',
                                'confidence': 'low'
                            }

                    # Создаем обогащенную запись
                    finding_entry = {
                        'question': str(question),
                        'summary': str(result.get('summary', '')),
                        'data_preview': data_preview[:10],  # Ограничиваем размер
                        'chart_data': convert_to_serializable(result.get('chart_data')),
                        'analyzed_tables': list(result.get('analyzed_tables', [])),
                        'method': 'smartgpt_dataframe_v2',
                        'analysis_type': analysis_type,

                        # SmartGPT обогащение
                        'smartgpt_insights': smartgpt_insights,
                        'has_gpt_insights': bool(smartgpt_insights.get('business_insights')),

                        # Дополнительные данные
                        'additional_info': convert_to_serializable(result.get('additional_info', {})),
                        'correlations': convert_to_serializable(result.get('correlations', [])),
                        'anomalies': convert_to_serializable(result.get('anomalies', [])),
                        'business_metrics': convert_to_serializable(result.get('business_metrics', {})),

                        'timestamp': datetime.now().isoformat(),
                        'success': True,
                        'smartgpt_enabled': enable_gpt
                    }

                    # Обновляем summary с SmartGPT инсайтами
                    if smartgpt_insights.get('business_insights'):
                        enhanced_summary = f"🤖 **SmartGPT Инсайты:**\n{smartgpt_insights['business_insights']}\n\n📊 **Технические данные:**\n{result.get('summary', '')}"
                        finding_entry['summary'] = enhanced_summary

                    smartgpt_findings.append(finding_entry)
                    successful_analyses += 1

                    logger.info(
                        f"[SMARTGPT DATAFRAME] ✅ Анализ {i + 1} завершен успешно (GPT: {'✨' if smartgpt_insights.get('business_insights') else '❌'})")

                else:
                    error_msg = str(result.get('error', 'Неизвестная ошибка'))
                    logger.error(f"[SMARTGPT DATAFRAME] ❌ Ошибка анализа {i + 1}: {error_msg}")

                    smartgpt_findings.append({
                        'question': str(question),
                        'summary': f'Ошибка анализа: {error_msg}',
                        'data_preview': [],
                        'error': error_msg,
                        'method': 'smartgpt_dataframe_v2',
                        'analysis_type': analysis_type,
                        'success': False,
                        'has_gpt_insights': False,
                        'timestamp': datetime.now().isoformat()
                    })

            except Exception as analysis_error:
                error_msg = f"Критическая ошибка SmartGPT анализа: {str(analysis_error)}"
                logger.error(f"[SMARTGPT DATAFRAME] 💥 {error_msg}")

                smartgpt_findings.append({
                    'question': str(question),
                    'summary': error_msg,
                    'data_preview': [],
                    'error': error_msg,
                    'method': 'smartgpt_dataframe_v2',
                    'analysis_type': analysis_type,
                    'success': False,
                    'has_gpt_insights': False,
                    'timestamp': datetime.now().isoformat()
                })

        # === СОЗДАНИЕ ФИНАЛЬНОГО SMARTGPT ОТЧЕТА ===
        self.update_state(
            state='FINALIZING',
            meta={'progress': '📝 Создание финального SmartGPT отчета...', 'progress_percentage': 85}
        )

        logger.info(
            f"[SMARTGPT DATAFRAME] 📊 Создание итогового отчета: {successful_analyses}/{len(smartgpt_findings)} успешных, {gpt_insights_count} SmartGPT инсайтов")

        # Получаем сводку по данным
        table_summary = df_manager.get_table_summary()
        memory_info = df_manager.get_memory_usage()

        # === SMARTGPT EXECUTIVE SUMMARY ===
        executive_summary = ""
        try:
            if successful_analyses > 0 and gpt_insights_count > 0:
                # Генерируем умное executive summary
                executive_summary = smart_gpt.generate_executive_summary(
                    smartgpt_findings, table_summary
                )
            else:
                executive_summary = _create_fallback_executive_summary(
                    smartgpt_findings, table_summary, successful_analyses
                )
        except Exception as summary_error:
            logger.error(f"Ошибка создания SmartGPT executive summary: {summary_error}")
            executive_summary = _create_fallback_executive_summary(
                smartgpt_findings, table_summary, successful_analyses
            )

        # === ФИНАЛЬНЫЙ SMARTGPT ОТЧЕТ ===
        final_smartgpt_report = {
            "executive_summary": str(executive_summary),
            "detailed_findings": convert_to_serializable(smartgpt_findings),
            "method": "smartgpt_dataframe_v2",
            "tables_info": convert_to_serializable(table_summary['tables']),
            "relations_info": convert_to_serializable(table_summary['relations']),

            "smartgpt_analysis_stats": {
                "questions_processed": int(len(smartgpt_findings)),
                "successful_analyses": int(successful_analyses),
                "failed_analyses": int(len(smartgpt_findings) - successful_analyses),
                "smartgpt_insights_count": int(gpt_insights_count),
                "tables_analyzed": int(table_summary['total_tables']),
                "relations_found": int(table_summary['total_relations']),
                "total_memory_mb": round(float(table_summary['total_memory_mb']), 2),
                "success_rate_percent": round(float(successful_analyses / max(len(smartgpt_findings), 1) * 100), 1),
                "smartgpt_coverage_percent": round(float(gpt_insights_count / max(successful_analyses, 1) * 100), 1)
            },

            "memory_usage": convert_to_serializable(memory_info),
            "smartgpt_recommendations": [str(r) for r in _generate_smartgpt_recommendations(
                smartgpt_findings, table_summary, successful_analyses, gpt_insights_count
            )],

            "report_metadata": {
                "created_at": datetime.now().isoformat(),
                "user_id": int(user_id),
                "connection_id": int(connection_id),
                "report_version": "3.0_smartgpt_dataframe",
                "max_questions_requested": int(max_questions),
                "smartgpt_enabled": True,
                "analysis_engine": "SmartGPTAnalyzer"
            }
        }

        # Проверяем размер отчета
        try:
            report_json = json.dumps(final_smartgpt_report, ensure_ascii=False)
            report_size_mb = len(report_json.encode('utf-8')) / 1024 / 1024
            logger.info(f"[SMARTGPT DATAFRAME] 📏 Размер SmartGPT отчета: {report_size_mb:.2f} MB")

            if report_size_mb > 12:  # Увеличенный лимит для SmartGPT отчетов
                logger.warning(f"[SMARTGPT DATAFRAME] ⚠️ Отчет слишком большой ({report_size_mb:.2f} MB), сокращаем")
                final_smartgpt_report = _trim_smartgpt_report(final_smartgpt_report)

        except Exception as json_error:
            logger.error(f"[SMARTGPT DATAFRAME] ❌ Ошибка JSON сериализации: {json_error}")
            final_smartgpt_report = convert_to_serializable(final_smartgpt_report)

        # === СОХРАНЕНИЕ SMARTGPT ОТЧЕТА ===
        self.update_state(
            state='SAVING',
            meta={'progress': '💾 Сохранение SmartGPT отчета в базу данных...', 'progress_percentage': 95}
        )

        try:
            crud.update_report(db_session, report_id, "COMPLETED", final_smartgpt_report)
            logger.info(f"[SMARTGPT DATAFRAME] ✅ SmartGPT отчет {report_id} успешно сохранен")

        except Exception as save_error:
            logger.error(f"[SMARTGPT DATAFRAME] ❌ Ошибка сохранения SmartGPT отчета: {save_error}")
            # Попытка сохранить упрощенный отчет
            try:
                simplified_report = {
                    "executive_summary": final_smartgpt_report["executive_summary"],
                    "method": "smartgpt_dataframe_v2",
                    "smartgpt_analysis_stats": final_smartgpt_report["smartgpt_analysis_stats"],
                    "error": f"Полный SmartGPT отчет не сохранен: {str(save_error)}",
                    "report_metadata": final_smartgpt_report["report_metadata"]
                }
                crud.update_report(db_session, report_id, "COMPLETED", simplified_report)
                logger.warning(f"[SMARTGPT DATAFRAME] ⚠️ Сохранен упрощенный SmartGPT отчет {report_id}")
            except Exception as final_save_error:
                logger.error(f"[SMARTGPT DATAFRAME] 💥 Критическая ошибка сохранения: {final_save_error}")
                raise

        self.update_state(
            state='SUCCESS',
            meta={'progress': '🎉 SmartGPT DataFrame анализ завершен успешно!', 'progress_percentage': 100}
        )

        return {
            "status": "success",
            "report_id": report_id,
            "questions_processed": len(smartgpt_findings),
            "successful_analyses": successful_analyses,
            "smartgpt_insights": gpt_insights_count,
            "tables_loaded": len(tables_loaded),
            "method": "smartgpt_dataframe_v2"
        }

    except Exception as e:
        logger.error(f"[SMARTGPT DATAFRAME ERROR] 💥 Критическая ошибка: {e}", exc_info=True)

        # Сохраняем информацию об ошибке
        error_report = {
            "error": str(e),
            "method": "smartgpt_dataframe_v2",
            "stage": "critical_error",
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "connection_id": connection_id,
            "smartgpt_enabled": True
        }

        try:
            crud.update_report(db_session, report_id, "FAILED", error_report)
        except Exception as save_error:
            logger.error(f"[SMARTGPT DATAFRAME] ❌ Не удалось сохранить ошибку: {save_error}")

        self.update_state(
            state='FAILURE',
            meta={
                'progress': f'💥 Критическая ошибка SmartGPT: {str(e)}',
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


def _create_smartgpt_analysis_plan(df_manager: DataFrameManager, max_questions: int) -> List[Dict[str, Any]]:
    """Создает интеллектуальный план анализа для SmartGPT"""

    plan = [
        {
            'question': '🏠 Умный обзор структуры данных с бизнес-контекстом и скрытыми возможностями',
            'type': 'overview',
            'enable_gpt': True,
            'priority': 1
        }
    ]

    # Анализ важных таблиц с SmartGPT приоритизацией
    tables_by_size = sorted(df_manager.tables.items(), key=lambda x: len(x[1]), reverse=True)

    for i, (table_name, df) in enumerate(tables_by_size[:min(4, len(tables_by_size))]):
        if i < 2:  # Первые 2 таблицы - детальный SmartGPT анализ
            plan.extend([
                {
                    'question': f"💼 Углубленный бизнес-анализ таблицы '{table_name}' с выявлением скрытых возможностей",
                    'type': 'business_insights',
                    'table_focus': table_name,
                    'enable_gpt': True,
                    'priority': 1 if i == 0 else 2
                },
                {
                    'question': f"🔍 Анализ качества данных таблицы '{table_name}' с умными рекомендациями по улучшению",
                    'type': 'data_quality',
                    'table_focus': table_name,
                    'enable_gpt': True,
                    'priority': 2
                }
            ])
        else:  # Остальные - экспресс-анализ
            plan.append({
                'question': f"⚡ Экспресс-анализ структуры таблицы '{table_name}' с ключевыми инсайтами",
                'type': 'table_analysis',
                'table_focus': table_name,
                'enable_gpt': True,
                'priority': 3
            })

    # Специализированные SmartGPT анализы на основе типов данных
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
                'question': '📊 Углубленный статистический анализ с SmartGPT интерпретацией паттернов',
                'type': 'statistical_insights',
                'enable_gpt': True,
                'priority': 2
            },
            {
                'question': '🔗 Анализ корреляций с бизнес-интерпретацией и практическими выводами',
                'type': 'correlation',
                'enable_gpt': True,
                'priority': 2
            }
        ])

    if has_datetime_data:
        plan.append({
            'question': '📈 Умный анализ временных трендов с предсказаниями и возможностями',
            'type': 'trend_analysis',
            'enable_gpt': True,
            'priority': 2
        })

    # Анализ связей между таблицами
    if df_manager.relations:
        plan.append({
            'question': '🌐 Анализ целостности связей с умными рекомендациями по оптимизации',
            'type': 'relationship_analysis',
            'enable_gpt': True,
            'priority': 2
        })

    # Поиск аномалий и возможностей
    plan.extend([
        {
            'question': '🚨 Умный поиск аномалий с анализом причин и влияния на бизнес',
            'type': 'anomalies',
            'enable_gpt': True,
            'priority': 3
        },
        {
            'question': '⚖️ Сравнительный анализ таблиц с выявлением скрытых паттернов',
            'type': 'comparison',
            'enable_gpt': True,
            'priority': 3
        }
    ])

    # Дополнительные SmartGPT анализы для больших планов
    if max_questions > 15:
        plan.extend([
            {
                'question': '🚀 Поиск скрытых бизнес-возможностей и точек роста через данные',
                'type': 'business_insights',
                'enable_gpt': True,
                'priority': 3
            },
            {
                'question': '🔮 Оценка потенциала для машинного обучения и автоматизации',
                'type': 'predictive_analysis',
                'enable_gpt': True,
                'priority': 4
            },
            {
                'question': '💡 Генерация инновационных идей на основе паттернов данных',
                'type': 'business_insights',
                'enable_gpt': True,
                'priority': 4
            }
        ])

    # Сортируем по приоритету и возвращаем нужное количество
    plan.sort(key=lambda x: x.get('priority', 5))
    return plan[:max_questions]


def _map_analysis_to_smartgpt_type(question: str, analysis_type: str) -> str:
    """Маппинг типов анализа для SmartGPT"""
    mapping = {
        'overview': 'business_insights',
        'table_analysis': 'business_insights',
        'business_insights': 'business_insights',
        'data_quality': 'data_quality',
        'statistical_insights': 'statistical_insights',
        'correlation': 'statistical_insights',
        'trend_analysis': 'predictive_analysis',
        'predictive_analysis': 'predictive_analysis',
        'relationship_analysis': 'data_quality',
        'anomalies': 'data_quality',
        'comparison': 'business_insights'
    }
    return mapping.get(analysis_type, 'business_insights')


def _extract_user_intent(question: str) -> str:
    """Извлекает намерение пользователя из вопроса"""
    question_lower = question.lower()

    if any(word in question_lower for word in ['проблем', 'ошибк', 'аномали', 'плохо']):
        return 'problem_solving'
    elif any(word in question_lower for word in ['возможност', 'потенциал', 'рост', 'улучшен']):
        return 'opportunity_discovery'
    elif any(word in question_lower for word in ['сравн', 'различ', 'vs', 'против']):
        return 'comparative_analysis'
    elif any(word in question_lower for word in ['тренд', 'динамик', 'прогноз', 'будущ']):
        return 'trend_analysis'
    elif any(word in question_lower for word in ['оптимиз', 'эффективност', 'производительност']):
        return 'optimization'
    else:
        return 'general_insights'


def _generate_smartgpt_recommendations(smartgpt_findings: List[dict], table_summary: dict,
                                       successful_analyses: int, gpt_insights_count: int) -> List[str]:
    """Генерирует SmartGPT рекомендации"""

    recommendations = []

    # Анализируем SmartGPT результаты
    all_action_items = []
    all_opportunities = []
    high_confidence_insights = 0

    for finding in smartgpt_findings:
        smartgpt_data = finding.get('smartgpt_insights', {})
        if smartgpt_data:
            if smartgpt_data.get('action_items'):
                all_action_items.extend(smartgpt_data['action_items'])
            if smartgpt_data.get('opportunities'):
                all_opportunities.extend(smartgpt_data['opportunities'])
            if smartgpt_data.get('confidence') == 'high':
                high_confidence_insights += 1

    # SmartGPT специфичные рекомендации
    if gpt_insights_count > 0:
        recommendations.append(
            f"🤖 Получено {gpt_insights_count} SmartGPT инсайтов с практическими рекомендациями"
        )

        if high_confidence_insights > gpt_insights_count * 0.7:
            recommendations.append(
                "✅ Высокое качество данных - SmartGPT инсайты имеют высокую достоверность"
            )

    # Приоритетные действия
    if all_action_items:
        unique_actions = list(set(all_action_items))[:3]
        recommendations.append(
            f"🎯 Выявлено {len(unique_actions)} приоритетных действий для немедленного внедрения"
        )

    # Бизнес-возможности
    if all_opportunities:
        unique_opportunities = list(set(all_opportunities))[:3]
        recommendations.append(
            f"🚀 Обнаружено {len(unique_opportunities)} возможностей для роста бизнеса"
        )

    # Автоматизация и мониторинг
    success_rate = (successful_analyses / max(len(smartgpt_findings), 1)) * 100
    gpt_coverage = (gpt_insights_count / max(successful_analyses, 1)) * 100

    if success_rate > 85 and gpt_coverage > 70:
        recommendations.extend([
            "🔄 Настройте регулярные SmartGPT отчеты для автоматического мониторинга",
            "📊 Создайте дашборды на основе выявленных SmartGPT метрик",
            "🔔 Внедрите систему алертов на основе найденных паттернов"
        ])

    # Развитие аналитики
    total_memory = table_summary.get('total_memory_mb', 0)
    if total_memory > 1000:
        recommendations.append(
            "⚡ Рассмотрите масштабирование для анализа еще больших данных"
        )

    recommendations.extend([
        "🎨 Визуализируйте SmartGPT инсайты для презентации руководству",
        "📈 Используйте выявленные паттерны для стратегического планирования",
        "🤝 Поделитесь бизнес-инсайтами с соответствующими командами",
        "🔮 Рассмотрите внедрение предиктивной аналитики на основе найденных трендов"
    ])

    return recommendations[:10]


def _create_fallback_executive_summary(smartgpt_findings: List[dict], table_summary: dict,
                                       successful_analyses: int) -> str:
    """Создает резервное executive summary"""

    total_questions = len(smartgpt_findings)
    total_tables = table_summary.get('total_tables', 0)
    total_relations = table_summary.get('total_relations', 0)

    return (
        f"Завершен SmartGPT DataFrame-анализ с {successful_analyses} успешными анализами "
        f"из {total_questions} запланированных. Проанализировано {total_tables} таблиц "
        f"с {total_relations} связями. SmartGPT инсайты генерируются автоматически."
    )


def _trim_smartgpt_report(report: dict) -> dict:
    """Сокращает размер SmartGPT отчета"""
    try:
        trimmed = report.copy()

        # Сокращаем detailed_findings
        if 'detailed_findings' in trimmed and isinstance(trimmed['detailed_findings'], list):
            for finding in trimmed['detailed_findings']:
                # Ограничиваем размер data_preview
                if 'data_preview' in finding and isinstance(finding['data_preview'], list):
                    if len(finding['data_preview']) > 3:
                        finding['data_preview'] = finding['data_preview'][:3]
                        finding['data_preview'].append({"note": "... данные сокращены для экономии места"})

                # Сокращаем длинные SmartGPT инсайты
                smartgpt_insights = finding.get('smartgpt_insights', {})
                if smartgpt_insights.get('business_insights') and len(smartgpt_insights['business_insights']) > 800:
                    smartgpt_insights['business_insights'] = smartgpt_insights['business_insights'][
                                                             :800] + "... (сокращено)"

        trimmed['report_metadata']['trimmed'] = True
        trimmed['report_metadata']['trim_reason'] = "SmartGPT отчет сокращен для оптимизации размера"

        return trimmed

    except Exception as e:
        logger.error(f"Ошибка сокращения SmartGPT отчета: {e}")
        return report


# =============== СПЕЦИАЛИЗИРОВАННЫЕ ЗАДАЧИ ===============

@celery_app.task(bind=True, time_limit=3600, name='tasks.quick_dataframe_analysis')
def quick_dataframe_analysis(self, connection_id: int, user_id: int, report_id: int):
    """Быстрый SmartGPT анализ (8 вопросов, ~10 минут)"""
    logger.info(f"[QUICK SMARTGPT] ⚡ Запуск для пользователя {user_id}")
    return generate_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 8}
    ).get()


@celery_app.task(bind=True, time_limit=9000, name='tasks.comprehensive_dataframe_analysis')
def comprehensive_dataframe_analysis(self, connection_id: int, user_id: int, report_id: int):
    """Комплексный SmartGPT анализ (25 вопросов, ~45 минут)"""
    logger.info(f"[COMPREHENSIVE SMARTGPT] 🧠 Запуск для пользователя {user_id}")
    return generate_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 25}
    ).get()


# =============== LEGACY ПОДДЕРЖКА ===============

@celery_app.task(bind=True, time_limit=7200, name='tasks.generate_advanced_report')
def generate_advanced_report(self, connection_id: int, user_id: int, report_id: int):
    """Legacy функция - перенаправляет на SmartGPT анализ"""
    logger.warning(f"[LEGACY] ⚠️ Перенаправление на SmartGPT анализ для пользователя {user_id}")
    return generate_dataframe_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 15}
    ).get()


logger.info("🚀 SmartGPT DataFrame Tasks система полностью загружена")
