# tasks.py

import pandas as pd
from celery.exceptions import Ignore
from sqlalchemy import create_engine
import crud
import database
from services.report_agents import (
    EnhancedOrchestrator,
    EnhancedSQLCoder,
    EnhancedCritic,
    MLPatternDetector,
    DomainAnalyzer,
    AdaptiveFeedbackSystem,
    AdvancedValidator,
    IntelligentPrioritizer,
    run_enhanced_analysis,
    get_database_health_check
)
from celery_worker import celery_app
import logging

# Настройка логирования для задач
logger = logging.getLogger(__name__)


@celery_app.task(bind=True, time_limit=7200, name='tasks.generate_enhanced_report')
def generate_enhanced_report(self, connection_id: int, user_id: int, report_id: int,
                             max_questions: int = 15, enable_feedback: bool = True):
    """
    Улучшенная Celery-задача для запуска продвинутого анализа с ML и адаптивностью.

    Args:
        connection_id: ID подключения к БД
        user_id: ID пользователя
        report_id: ID отчета
        max_questions: Максимальное количество вопросов для анализа
        enable_feedback: Включить систему обратной связи
    """

    db_session = next(database.get_db())

    try:
        logger.info(f"[ENHANCED TASK START] Report ID: {report_id}, User ID: {user_id}")

        # === ЭТАП 1: ИНИЦИАЛИЗАЦИЯ ===
        self.update_state(
            state='INITIALIZING',
            meta={'progress': 'Инициализация улучшенной системы анализа...', 'stage': 'setup'}
        )

        # Получаем строку подключения
        connection_string = crud.get_decrypted_connection_string(db_session, connection_id, user_id)
        if not connection_string:
            crud.update_report(db_session, report_id, "FAILED", {
                "error": "Подключение к БД не найдено.",
                "stage": "initialization"
            })
            raise Ignore()

        logger.info("[ENHANCED TASK] Connection string retrieved successfully")

        # Создаем подключение к БД
        engine = create_engine(connection_string, connect_args={'connect_timeout': 30})

        # === ЭТАП 2: ПРОВЕРКА ЗДОРОВЬЯ БД ===
        self.update_state(
            state='HEALTH_CHECK',
            meta={'progress': 'Проверка состояния базы данных...', 'stage': 'health_check'}
        )

        health_check = get_database_health_check(engine)

        if not health_check["connection"]:
            crud.update_report(db_session, report_id, "FAILED", {
                "error": "Не удалось подключиться к базе данных",
                "health_check": health_check,
                "stage": "health_check"
            })
            raise Ignore()

        if not health_check["has_data"]:
            crud.update_report(db_session, report_id, "FAILED", {
                "error": "База данных не содержит данных",
                "health_check": health_check,
                "stage": "health_check"
            })
            raise Ignore()

        logger.info(f"[ENHANCED TASK] Database health check passed. Total rows: {health_check['total_rows']}")

        # === ЭТАП 3: ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ ===
        self.update_state(
            state='COMPONENT_INIT',
            meta={'progress': 'Инициализация ML-компонентов...', 'stage': 'component_init'}
        )

        # Инициализируем улучшенные компоненты
        orchestrator = EnhancedOrchestrator(engine)
        sql_coder = EnhancedSQLCoder(engine)
        critic = EnhancedCritic(orchestrator.ml_detector)

        logger.info("[ENHANCED TASK] All enhanced components initialized")

        # === ЭТАП 4: СОЗДАНИЕ ИНТЕЛЛЕКТУАЛЬНОГО ПЛАНА ===
        self.update_state(
            state='PLANNING',
            meta={'progress': 'Создание интеллектуального плана анализа...', 'stage': 'planning'}
        )

        # Создаем интеллектуальный план с ML и контекстным пониманием
        analysis_plan = orchestrator.create_intelligent_plan()

        logger.info(f"[ENHANCED TASK] Intelligent plan created with {len(analysis_plan)} questions")

        # Получаем информацию о домене
        schema = orchestrator.get_comprehensive_schema()
        populated_tables = {k: v for k, v in schema.items() if v.get('row_count', 0) > 0}
        domain_context = orchestrator.domain_analyzer.detect_domain(
            list(populated_tables.keys()), schema
        )

        logger.info(f"[ENHANCED TASK] Domain detected: {domain_context.domain_type} "
                    f"(confidence: {domain_context.confidence:.2f})")

        # === ЭТАП 5: ВЫПОЛНЕНИЕ АНАЛИЗА ===
        session_memory = []
        questions_processed = 0
        ml_insights_collected = []

        for question in analysis_plan:
            if questions_processed >= max_questions:
                break

            questions_processed += 1
            progress_percentage = (questions_processed / min(max_questions, len(analysis_plan))) * 100

            self.update_state(
                state='ANALYZING',
                meta={
                    'progress': f"Анализ {questions_processed}/{max_questions}: {question[:50]}...",
                    'stage': 'analysis',
                    'question': question,
                    'progress_percentage': progress_percentage
                }
            )

            logger.info(f"[ENHANCED TASK] Processing question {questions_processed}/{max_questions}: {question}")

            # SQL Coder с валидацией
            execution_result = sql_coder.run_with_validation(question)

            # Critic с ML-анализом
            evaluation = critic.evaluate_with_ml(execution_result)

            # Логируем результаты
            logger.info(f"[ENHANCED TASK] Question processed. Success: {evaluation.get('is_success')}")

            if evaluation.get('finding'):
                finding = evaluation['finding']
                session_memory.append(finding)

                # Собираем ML-инсайты
                ml_patterns = finding.get('ml_patterns', [])
                if ml_patterns:
                    ml_insights_collected.extend(ml_patterns)

                # Симулируем пользовательскую обратную связь (в реальном приложении можно получать от UI)
                if enable_feedback:
                    simulated_rating = 4 if evaluation.get('is_success') else 2
                    orchestrator.process_evaluation_with_feedback(
                        evaluation, session_memory, analysis_plan, question, simulated_rating
                    )
                else:
                    orchestrator.process_evaluation_with_feedback(
                        evaluation, session_memory, analysis_plan, question
                    )

            # Периодически обновляем состояние с промежуточными результатами
            if questions_processed % 3 == 0:
                diversity_report = orchestrator.get_analysis_diversity_report(session_memory)
                self.update_state(
                    state='ANALYZING',
                    meta={
                        'progress': f"Завершено {questions_processed} вопросов. "
                                    f"Покрытие таблиц: {diversity_report['coverage_percentage']:.1f}%",
                        'stage': 'analysis',
                        'diversity_report': diversity_report,
                        'progress_percentage': progress_percentage
                    }
                )

        # === ЭТАП 6: СОЗДАНИЕ ФИНАЛЬНОГО ОТЧЕТА ===
        self.update_state(
            state='SYNTHESIZING',
            meta={'progress': 'Создание комплексного отчета...', 'stage': 'synthesis'}
        )

        logger.info(f"[ENHANCED TASK] Creating final report with {len(session_memory)} findings")

        if not session_memory:
            final_report = {
                "executive_summary": "Анализ не дал результатов. База данных может быть пустой или недоступной.",
                "detailed_findings": [],
                "health_check": health_check,
                "domain_context": {
                    "domain_type": domain_context.domain_type,
                    "confidence": domain_context.confidence
                },
                "analysis_stats": {
                    "questions_processed": questions_processed,
                    "ml_patterns_found": 0,
                    "tables_analyzed": 0
                }
            }
        else:
            # Создаем комплексный отчет с ML-инсайтами
            final_report = create_comprehensive_report(
                session_memory,
                orchestrator,
                health_check,
                domain_context,
                questions_processed,
                ml_insights_collected,
                enable_feedback
            )

        # === ЭТАП 7: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
        self.update_state(
            state='SAVING',
            meta={'progress': 'Сохранение отчета...', 'stage': 'saving'}
        )

        crud.update_report(db_session, report_id, "COMPLETED", final_report)

        logger.info("[ENHANCED TASK] Report saved successfully")

        self.update_state(
            state='SUCCESS',
            meta={
                'progress': 'Анализ завершен!',
                'stage': 'completed',
                'summary': {
                    'questions_processed': questions_processed,
                    'findings_count': len(session_memory),
                    'ml_patterns_found': len(ml_insights_collected),
                    'domain_detected': domain_context.domain_type
                }
            }
        )

    except Exception as e:
        logger.error(f"[ENHANCED TASK ERROR] {e}", exc_info=True)

        # Детальная информация об ошибке
        error_report = {
            "error": "Произошла критическая ошибка в улучшенной задаче анализа",
            "details": str(e),
            "stage": "unknown",
            "type": type(e).__name__
        }

        crud.update_report(db_session, report_id, "FAILED", error_report)

        self.update_state(
            state='FAILURE',
            meta={
                'progress': 'Ошибка выполнения',
                'error': str(e),
                'stage': 'error'
            }
        )

        raise e

    finally:
        db_session.close()


def create_comprehensive_report(session_memory, orchestrator, health_check,
                                domain_context, questions_processed, ml_insights,
                                enable_feedback):
    """
    Создает комплексный отчет с ML-инсайтами и адаптивной аналитикой.
    """

    # Получаем ML-инсайты от оркестратора
    ml_summary = orchestrator.get_ml_insights_summary()

    # Получаем отчет о разнообразии анализа
    diversity_report = orchestrator.get_analysis_diversity_report(session_memory)

    # Создаем executive summary
    executive_summary = generate_executive_summary(
        session_memory, ml_summary, domain_context, questions_processed
    )

    # Формируем детальные результаты
    detailed_findings = []
    for finding in session_memory:
        # Обогащаем каждый результат дополнительной информацией
        enhanced_finding = finding.copy()
        enhanced_finding['timestamp'] = finding.get('timestamp', 'N/A')
        enhanced_finding['confidence_score'] = calculate_confidence_score(finding)
        detailed_findings.append(enhanced_finding)

    # Создаем рекомендации
    recommendations = generate_smart_recommendations(
        session_memory, ml_summary, domain_context, diversity_report
    )

    # Формируем финальный отчет
    final_report = {
        "executive_summary": executive_summary,
        "detailed_findings": detailed_findings,
        "recommendations": recommendations,
        "health_check": health_check,
        "domain_context": {
            "domain_type": domain_context.domain_type,
            "confidence": domain_context.confidence,
            "key_entities": domain_context.key_entities,
            "business_metrics": domain_context.business_metrics
        },
        "ml_insights": ml_summary,
        "analysis_stats": {
            "questions_processed": questions_processed,
            "successful_findings": len([f for f in session_memory if f.get('data_preview')]),
            "ml_patterns_found": len(ml_insights),
            "tables_coverage": diversity_report['coverage_percentage'],
            "tables_analyzed": diversity_report['analyzed_tables']
        },
        "diversity_report": diversity_report
    }

    # Добавляем адаптивную стратегию если обратная связь включена
    if enable_feedback:
        final_report["adaptive_strategy"] = orchestrator.feedback_system.adapt_analysis_strategy()

    return final_report


def generate_executive_summary(session_memory, ml_summary, domain_context, questions_processed):
    """Генерирует executive summary с учетом ML-инсайтов."""

    successful_findings = len([f for f in session_memory if f.get('data_preview')])
    charts_created = len([f for f in session_memory if f.get('chart_url')])

    summary = f"Проведен интеллектуальный анализ {questions_processed} вопросов с использованием машинного обучения. "
    summary += f"Получено {successful_findings} успешных результатов, создано {charts_created} визуализаций. "

    if domain_context.domain_type != 'general':
        summary += f"Определена предметная область: {domain_context.domain_type} "
        summary += f"(уверенность: {domain_context.confidence:.1%}). "

    if ml_summary.get('total_patterns', 0) > 0:
        summary += f"Обнаружено {ml_summary['total_patterns']} паттернов с помощью ML-алгоритмов. "

    return summary


def calculate_confidence_score(finding):
    """Вычисляет оценку уверенности для результата."""

    score = 0.5  # Базовая оценка

    # Увеличиваем за наличие данных
    if finding.get('data_preview'):
        score += 0.2
        if len(finding['data_preview']) > 10:
            score += 0.1

    # Увеличиваем за наличие визуализации
    if finding.get('chart_url'):
        score += 0.1

    # Увеличиваем за ML-паттерны
    if finding.get('ml_patterns'):
        high_conf_patterns = [p for p in finding['ml_patterns'] if p.get('confidence', 0) > 0.7]
        if high_conf_patterns:
            score += 0.1

    # Увеличиваем за успешную валидацию
    if finding.get('validation', {}).get('is_valid'):
        score += 0.1

    return min(score, 1.0)


def generate_smart_recommendations(session_memory, ml_summary, domain_context, diversity_report):
    """Генерирует умные рекомендации на основе результатов анализа."""

    recommendations = []

    # Рекомендации на основе покрытия таблиц
    if diversity_report['coverage_percentage'] < 70:
        recommendations.append(
            f"Рекомендуется провести дополнительный анализ недоисследованных таблиц: "
            f"{', '.join(diversity_report['underanalyzed_tables'][:3])}"
        )

    # Рекомендации на основе ML-паттернов
    if ml_summary.get('total_patterns', 0) > 0:
        pattern_types = list(ml_summary.get('pattern_types', {}).keys())
        if 'anomaly' in pattern_types:
            recommendations.append("Внедрить систему мониторинга для автоматического обнаружения аномалий")
        if 'clustering' in pattern_types:
            recommendations.append("Провести бизнес-интерпретацию обнаруженных кластеров")
        if 'correlation' in pattern_types:
            recommendations.append("Исследовать причинно-следственные связи в обнаруженных корреляциях")

    # Рекомендации на основе предметной области
    if domain_context.domain_type == 'ecommerce':
        recommendations.extend([
            "Создать дашборд для мониторинга ключевых метрик продаж",
            "Внедрить систему сегментации клиентов"
        ])
    elif domain_context.domain_type == 'crm':
        recommendations.extend([
            "Оптимизировать воронку продаж на основе выявленных паттернов",
            "Автоматизировать скоринг лидов"
        ])
    elif domain_context.domain_type == 'analytics':
        recommendations.extend([
            "Настроить A/B тестирование для оптимизации конверсий",
            "Создать когортный анализ пользователей"
        ])

    # Общие рекомендации
    recommendations.extend([
        "Автоматизировать регулярное обновление анализа",
        "Создать систему уведомлений о важных изменениях в данных"
    ])

    return recommendations[:8]  # Ограничиваем количество рекомендаций


@celery_app.task(bind=True, time_limit=1800, name='tasks.quick_ml_analysis')
def quick_ml_analysis(self, connection_id: int, user_id: int, report_id: int):
    """
    Быстрый ML-анализ для оперативного обнаружения паттернов.
    """

    db_session = next(database.get_db())

    try:
        self.update_state(state='INITIALIZING', meta={'progress': 'Инициализация быстрого ML-анализа...'})

        connection_string = crud.get_decrypted_connection_string(db_session, connection_id, user_id)
        if not connection_string:
            raise Ignore()

        engine = create_engine(connection_string, connect_args={'connect_timeout': 20})

        # Используем функцию быстрого анализа
        result = run_enhanced_analysis(engine, max_questions=8, enable_feedback=False)

        crud.update_report(db_session, report_id, "COMPLETED", result)

        self.update_state(state='SUCCESS', meta={'progress': 'Быстрый анализ завершен!'})

    except Exception as e:
        logger.error(f"[QUICK ML ANALYSIS ERROR] {e}")
        crud.update_report(db_session, report_id, "FAILED", {"error": str(e)})
        raise e
    finally:
        db_session.close()


# Обратная совместимость со старой функцией
@celery_app.task(bind=True, time_limit=3600, name='tasks.generate_advanced_report')
def generate_advanced_report(self, connection_id: int, user_id: int, report_id: int):
    """
    Обратная совместимость: перенаправляет на новую улучшенную функцию.
    """
    return generate_enhanced_report.apply_async(
        args=[connection_id, user_id, report_id],
        kwargs={'max_questions': 12, 'enable_feedback': True}
    )
