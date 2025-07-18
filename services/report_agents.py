import ast
import io
import json
import uuid
import logging
import re
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import boto3
import openai
import plotly.express as px
import plotly.graph_objects as go
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
import config

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Клиенты и утилиты
s3_client = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_DEFAULT_REGION
)

openai_client = openai.OpenAI(api_key=config.API_KEY)


# ================== МАШИННОЕ ОБУЧЕНИЕ ==================

@dataclass
class DataPattern:
    """Структура для хранения обнаруженных паттернов"""
    pattern_type: str
    description: str
    confidence: float
    tables_involved: List[str]
    columns_involved: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Конвертирует DataPattern в сериализуемый словарь"""
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "confidence": float(self.confidence),
            "tables_involved": self.tables_involved,
            "columns_involved": self.columns_involved,
            "metadata": self._serialize_metadata(self.metadata),
            "timestamp": self.timestamp.isoformat()
        }

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Сериализует метаданные для JSON"""
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, (list, np.ndarray)):
                # Конвертируем numpy массивы в обычные списки
                serialized[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
            elif isinstance(value, (np.floating, np.integer)):
                serialized[key] = float(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = value
        return serialized


class MLPatternDetector:
    """Система машинного обучения для обнаружения паттернов в данных"""

    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.clustering_model = KMeans(n_clusters=3, random_state=42)
        self.detected_patterns = []

    def detect_anomalies(self, df: pd.DataFrame, table_name: str) -> List[DataPattern]:
        """Обнаруживает аномалии в числовых данных"""
        patterns = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return patterns

        try:
            # Подготавливаем данные
            numeric_data = df[numeric_cols].fillna(0)
            if len(numeric_data) < 2:
                return patterns

            # Масштабируем данные
            scaled_data = self.scaler.fit_transform(numeric_data)

            # Обнаруживаем аномалии
            anomaly_scores = self.anomaly_detector.fit_predict(scaled_data)
            anomaly_indices = np.where(anomaly_scores == -1)[0]

            if len(anomaly_indices) > 0:
                anomaly_percentage = len(anomaly_indices) / len(df) * 100

                pattern = DataPattern(
                    pattern_type="anomaly",
                    description=f"Обнаружено {len(anomaly_indices)} аномальных записей ({anomaly_percentage:.1f}%) в таблице {table_name}",
                    confidence=0.8,
                    tables_involved=[table_name],
                    columns_involved=list(numeric_cols),
                    metadata={
                        "anomaly_count": len(anomaly_indices),
                        "anomaly_percentage": anomaly_percentage,
                        "anomaly_indices": anomaly_indices.tolist()
                    }
                )
                patterns.append(pattern)

        except Exception as e:
            logger.error(f"Ошибка обнаружения аномалий: {e}")

        return patterns

    def detect_clusters(self, df: pd.DataFrame, table_name: str) -> List[DataPattern]:
        """Обнаруживает кластеры в данных"""
        patterns = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return patterns

        try:
            # Подготавливаем данные
            numeric_data = df[numeric_cols].fillna(0)
            if len(numeric_data) < 3:
                return patterns

            # Масштабируем данные
            scaled_data = self.scaler.fit_transform(numeric_data)

            # Определяем оптимальное количество кластеров
            max_clusters = min(5, len(numeric_data) // 2)
            if max_clusters < 2:
                return patterns

            self.clustering_model.n_clusters = max_clusters
            cluster_labels = self.clustering_model.fit_predict(scaled_data)

            # Анализируем кластеры
            unique_clusters = np.unique(cluster_labels)
            if len(unique_clusters) > 1:
                cluster_sizes = [np.sum(cluster_labels == i) for i in unique_clusters]

                pattern = DataPattern(
                    pattern_type="clustering",
                    description=f"Обнаружено {len(unique_clusters)} кластеров в таблице {table_name}",
                    confidence=0.7,
                    tables_involved=[table_name],
                    columns_involved=list(numeric_cols),
                    metadata={
                        "cluster_count": len(unique_clusters),
                        "cluster_sizes": cluster_sizes,
                        "cluster_labels": cluster_labels.tolist()
                    }
                )
                patterns.append(pattern)

        except Exception as e:
            logger.error(f"Ошибка кластеризации: {e}")

        return patterns

    def detect_correlations(self, df: pd.DataFrame, table_name: str) -> List[DataPattern]:
        """Обнаруживает сильные корреляции между переменными"""
        patterns = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return patterns

        try:
            # Вычисляем корреляционную матрицу
            corr_matrix = df[numeric_cols].corr()

            # Ищем сильные корреляции
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # Сильная корреляция
                        strong_correlations.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })

            if strong_correlations:
                pattern = DataPattern(
                    pattern_type="correlation",
                    description=f"Обнаружено {len(strong_correlations)} сильных корреляций в таблице {table_name}",
                    confidence=0.9,
                    tables_involved=[table_name],
                    columns_involved=list(numeric_cols),
                    metadata={
                        "correlations": strong_correlations
                    }
                )
                patterns.append(pattern)

        except Exception as e:
            logger.error(f"Ошибка анализа корреляций: {e}")

        return patterns

    def detect_all_patterns(self, df: pd.DataFrame, table_name: str) -> List[DataPattern]:
        """Обнаруживает все типы паттернов"""
        all_patterns = []

        all_patterns.extend(self.detect_anomalies(df, table_name))
        all_patterns.extend(self.detect_clusters(df, table_name))
        all_patterns.extend(self.detect_correlations(df, table_name))

        # Сохраняем обнаруженные паттерны
        self.detected_patterns.extend(all_patterns)

        return all_patterns


# ================== КОНТЕКСТНОЕ ПОНИМАНИЕ ==================

@dataclass
class DomainContext:
    """Контекст предметной области"""
    domain_type: str
    confidence: float
    key_entities: List[str]
    relationships: Dict[str, List[str]]
    business_metrics: List[str]


class DomainAnalyzer:
    """Анализатор предметной области"""

    def __init__(self):
        self.domain_patterns = {
            'ecommerce': {
                'tables': ['orders', 'products', 'customers', 'payments', 'cart', 'inventory'],
                'relationships': ['customer_orders', 'order_products', 'product_categories'],
                'metrics': ['revenue', 'conversion_rate', 'average_order_value', 'customer_lifetime_value']
            },
            'crm': {
                'tables': ['contacts', 'leads', 'deals', 'activities', 'companies', 'users'],
                'relationships': ['contact_activities', 'deal_contacts', 'company_contacts'],
                'metrics': ['conversion_rate', 'deal_value', 'pipeline_velocity', 'customer_acquisition_cost']
            },
            'analytics': {
                'tables': ['events', 'sessions', 'users', 'metrics', 'dimensions', 'pageviews'],
                'relationships': ['user_sessions', 'session_events', 'event_metrics'],
                'metrics': ['bounce_rate', 'session_duration', 'page_views', 'unique_visitors']
            },
            'finance': {
                'tables': ['transactions', 'accounts', 'balances', 'invoices', 'payments'],
                'relationships': ['account_transactions', 'invoice_payments', 'transaction_categories'],
                'metrics': ['total_revenue', 'profit_margin', 'cash_flow', 'account_balance']
            },
            'hr': {
                'tables': ['employees', 'departments', 'positions', 'salaries', 'performance'],
                'relationships': ['employee_departments', 'department_positions', 'employee_performance'],
                'metrics': ['turnover_rate', 'average_salary', 'performance_rating', 'headcount']
            }
        }

        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    def detect_domain(self, table_names: List[str], schema_info: Dict) -> DomainContext:
        """Определяет предметную область на основе названий таблиц"""

        domain_scores = {}

        for domain, patterns in self.domain_patterns.items():
            score = 0
            matched_tables = []

            # Проверяем совпадения названий таблиц
            for table in table_names:
                table_lower = table.lower()
                for pattern_table in patterns['tables']:
                    if pattern_table in table_lower or table_lower in pattern_table:
                        score += 1
                        matched_tables.append(table)
                        break

            # Нормализуем счет
            if len(patterns['tables']) > 0:
                normalized_score = score / len(patterns['tables'])
                domain_scores[domain] = {
                    'score': normalized_score,
                    'matched_tables': matched_tables
                }

        # Выбираем домен с максимальным счетом
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1]['score'])
            domain_name = best_domain[0]
            domain_data = best_domain[1]

            if domain_data['score'] > 0.2:  # Минимальный порог уверенности
                return DomainContext(
                    domain_type=domain_name,
                    confidence=domain_data['score'],
                    key_entities=domain_data['matched_tables'],
                    relationships=self.domain_patterns[domain_name]['relationships'],
                    business_metrics=self.domain_patterns[domain_name]['metrics']
                )

        # Возвращаем общий контекст если не удалось определить домен
        return DomainContext(
            domain_type='general',
            confidence=0.5,
            key_entities=table_names,
            relationships=[],
            business_metrics=['count', 'average', 'sum', 'min', 'max']
        )

    def generate_domain_questions(self, context: DomainContext, schema_info: Dict) -> List[str]:
        """Генерирует вопросы, специфичные для предметной области"""

        questions = []

        if context.domain_type == 'ecommerce':
            questions.extend([
                "Какие товары приносят наибольшую прибыль?",
                "Как изменилась динамика продаж за последний период?",
                "Какие клиенты самые ценные по объему покупок?",
                "Есть ли сезонные тренды в продажах?",
                "Какие категории товаров популярны?",
                "Анализ конверсии от просмотра к покупке"
            ])

        elif context.domain_type == 'crm':
            questions.extend([
                "Какие лиды имеют наибольшую вероятность конверсии?",
                "Как долго в среднем проходит сделка по воронке?",
                "Какие активности наиболее эффективны для закрытия сделок?",
                "Анализ производительности менеджеров по продажам",
                "Сегментация клиентов по активности",
                "Прогноз выполнения плана продаж"
            ])

        elif context.domain_type == 'analytics':
            questions.extend([
                "Какие страницы имеют наибольшую посещаемость?",
                "Анализ поведения пользователей на сайте",
                "Какие источники трафика наиболее эффективны?",
                "Время проведенное пользователями на сайте",
                "Анализ конверсии по воронке",
                "Сегментация пользователей по поведению"
            ])

        # Добавляем вопросы на основе ключевых сущностей
        for entity in context.key_entities[:3]:  # Берем первые 3 сущности
            questions.append(f"Детальный анализ данных в таблице {entity}")
            questions.append(f"Временные тренды в таблице {entity}")

        return questions


# ================== СИСТЕМА ОБРАТНОЙ СВЯЗИ ==================

@dataclass
class FeedbackEntry:
    """Запись обратной связи"""
    question: str
    finding: Dict[str, Any]
    rating: int  # 1-5
    feedback_text: str
    timestamp: datetime = field(default_factory=datetime.now)


class AdaptiveFeedbackSystem:
    """Система адаптивной обратной связи"""

    def __init__(self):
        self.feedback_history = []
        self.user_preferences = {}
        self.question_success_rates = {}
        self.learned_patterns = {}

    def collect_feedback(self, question: str, finding: Dict[str, Any], rating: int, feedback_text: str = ""):
        """Собирает обратную связь от пользователя"""

        feedback = FeedbackEntry(
            question=question,
            finding=finding,
            rating=rating,
            feedback_text=feedback_text
        )

        self.feedback_history.append(feedback)

        # Обновляем статистику успешности вопросов
        self._update_question_success_rates(question, rating)

        # Обновляем предпочтения пользователя
        self._update_user_preferences(finding, rating)

        logger.info(f"Получена обратная связь: {rating}/5 для вопроса: {question[:50]}...")

    def _update_question_success_rates(self, question: str, rating: int):
        """Обновляет статистику успешности типов вопросов"""

        # Определяем тип вопроса
        question_type = self._categorize_question(question)

        if question_type not in self.question_success_rates:
            self.question_success_rates[question_type] = []

        self.question_success_rates[question_type].append(rating)

    def _update_user_preferences(self, finding: Dict[str, Any], rating: int):
        """Обновляет предпочтения пользователя"""

        # Анализируем характеристики понравившихся результатов
        if rating >= 4:  # Хорошая оценка
            chart_url = finding.get('chart_url')
            data_preview = finding.get('data_preview', [])

            if chart_url:
                self.user_preferences['likes_charts'] = self.user_preferences.get('likes_charts', 0) + 1

            # Безопасная проверка data_preview
            if data_preview and isinstance(data_preview, list) and len(data_preview) > 20:
                self.user_preferences['likes_detailed_data'] = self.user_preferences.get('likes_detailed_data', 0) + 1

            # Анализируем типы данных
            if finding.get('data_stats'):
                stats = finding['data_stats']
                if stats.get('numeric_stats'):
                    self.user_preferences['likes_numeric_analysis'] = self.user_preferences.get(
                        'likes_numeric_analysis', 0) + 1
                if stats.get('categorical_stats'):
                    self.user_preferences['likes_categorical_analysis'] = self.user_preferences.get(
                        'likes_categorical_analysis', 0) + 1

    def _categorize_question(self, question: str) -> str:
        """Категоризирует вопрос по типу"""

        question_lower = question.lower()

        if any(keyword in question_lower for keyword in ['тренд', 'динамика', 'время', 'дата']):
            return 'temporal'
        elif any(keyword in question_lower for keyword in ['корреляция', 'связь', 'зависимость']):
            return 'correlation'
        elif any(keyword in question_lower for keyword in ['аномалия', 'выброс', 'отклонение']):
            return 'anomaly'
        elif any(keyword in question_lower for keyword in ['группировка', 'сегментация', 'категория']):
            return 'segmentation'
        else:
            return 'general'

    def get_preferred_question_types(self) -> List[str]:
        """Возвращает предпочтительные типы вопросов на основе обратной связи"""

        if not self.question_success_rates:
            return ['general', 'temporal', 'correlation']

        # Вычисляем средние рейтинги для каждого типа
        average_ratings = {}
        for question_type, ratings in self.question_success_rates.items():
            average_ratings[question_type] = np.mean(ratings)

        # Сортируем по убыванию среднего рейтинга
        sorted_types = sorted(average_ratings.items(), key=lambda x: x[1], reverse=True)

        return [qtype for qtype, _ in sorted_types]

    def adapt_analysis_strategy(self) -> Dict[str, Any]:
        """Адаптирует стратегию анализа на основе обратной связи"""

        strategy = {
            'preferred_question_types': self.get_preferred_question_types(),
            'generate_charts': self.user_preferences.get('likes_charts', 0) > 2,
            'detailed_data': self.user_preferences.get('likes_detailed_data', 0) > 2,
            'focus_numeric': self.user_preferences.get('likes_numeric_analysis', 0) > 2,
            'focus_categorical': self.user_preferences.get('likes_categorical_analysis', 0) > 2
        }

        return strategy


# ================== УЛУЧШЕННАЯ ВАЛИДАЦИЯ ==================

class AdvancedValidator:
    """Продвинутая система валидации SQL и данных"""

    def __init__(self, engine):
        self.engine = engine
        self.inspector = inspect(engine)
        self.table_names = [t for t in self.inspector.get_table_names() if t != 'alembic_version']

    def validate_sql_query(self, sql_query: str) -> Tuple[bool, str]:
        """Расширенная валидация SQL запроса"""

        if not sql_query or not sql_query.strip():
            return False, "Пустой SQL запрос"

        sql_lower = sql_query.lower().strip()

        # Проверка на наличие обязательных элементов
        if not sql_lower.startswith('select'):
            return False, "Запрос должен начинаться с SELECT"

        if 'from' not in sql_lower:
            return False, "Запрос должен содержать FROM"

        # Проверка на опасные операции
        dangerous_operations = ['drop', 'delete', 'update', 'insert', 'alter', 'create']
        if any(op in sql_lower for op in dangerous_operations):
            return False, f"Запрос содержит опасную операцию: {sql_query}"

        # Проверка названий таблиц
        tables_in_query = self._extract_tables_from_sql(sql_query)
        invalid_tables = [t for t in tables_in_query if t not in self.table_names]

        if invalid_tables:
            return False, f"Неизвестные таблицы: {', '.join(invalid_tables)}"

        # Проверка синтаксиса через dry run
        try:
            with self.engine.connect() as conn:
                # Добавляем LIMIT 0 для проверки синтаксиса без выполнения
                test_query = f"SELECT * FROM ({sql_query}) as test_query LIMIT 0"
                conn.execute(text(test_query))
            return True, "Валидация прошла успешно"
        except Exception as e:
            return False, f"Синтаксическая ошибка: {str(e)}"

    def _extract_tables_from_sql(self, sql_query: str) -> List[str]:
        """Извлекает названия таблиц из SQL запроса"""

        tables = []
        sql_lower = sql_query.lower()

        # Паттерны для поиска таблиц
        patterns = [
            r'from\s+(\w+)',
            r'join\s+(\w+)',
            r'into\s+(\w+)',
            r'update\s+(\w+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, sql_lower)
            tables.extend(matches)

        return list(set(tables))

    def validate_data_quality(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Валидирует качество полученных данных"""

        quality_report = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }

        if df.empty:
            quality_report['is_valid'] = False
            quality_report['issues'].append("Нет данных")
            return quality_report

        # Проверка на пропущенные значения
        null_counts = df.isnull().sum()
        high_null_cols = null_counts[null_counts > len(df) * 0.5].index.tolist()

        if high_null_cols:
            quality_report['warnings'].append(f"Много пропущенных значений в колонках: {', '.join(high_null_cols)}")

        # Проверка на дубликаты
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            quality_report['warnings'].append(f"Найдено {duplicate_count} дубликатов")

        # Статистика качества
        quality_report['stats'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_count': duplicate_count
        }

        return quality_report


# ================== ИНТЕЛЛЕКТУАЛЬНАЯ ПРИОРИТИЗАЦИЯ ==================

class IntelligentPrioritizer:
    """Интеллектуальная система приоритизации анализа"""

    def __init__(self, engine):
        self.engine = engine
        self.inspector = inspect(engine)
        self.table_graph = self._build_table_graph()
        self.analysis_history = {}

    def _build_table_graph(self) -> nx.DiGraph:
        """Строит граф связей между таблицами"""

        graph = nx.DiGraph()

        # Добавляем все таблицы как узлы
        for table in self.inspector.get_table_names():
            if table != 'alembic_version':
                graph.add_node(table)

        # Добавляем связи на основе внешних ключей
        for table in graph.nodes():
            try:
                foreign_keys = self.inspector.get_foreign_keys(table)
                for fk in foreign_keys:
                    referred_table = fk.get('referred_table')
                    if referred_table and referred_table in graph.nodes():
                        graph.add_edge(table, referred_table)
            except Exception as e:
                logger.warning(f"Не удалось получить внешние ключи для {table}: {e}")

        return graph

    def calculate_table_importance(self, schema_info: Dict) -> Dict[str, float]:
        """Вычисляет важность каждой таблицы"""

        importance_scores = {}

        for table in self.table_graph.nodes():
            score = 0.0

            # Базовая важность на основе размера
            row_count = schema_info.get(table, {}).get('row_count', 0)
            if row_count > 0:
                score += min(np.log10(row_count), 5)  # Логарифмическая шкала, максимум 5

            # Важность на основе связей (центральность)
            try:
                # Входящие связи (на сколько таблиц ссылается)
                in_degree = self.table_graph.in_degree(table)
                score += in_degree * 0.5

                # Исходящие связи (сколько таблиц ссылается на эту)
                out_degree = self.table_graph.out_degree(table)
                score += out_degree * 0.3

                # Центральность по промежуточности
                betweenness = nx.betweenness_centrality(self.table_graph).get(table, 0)
                score += betweenness * 2

            except Exception as e:
                logger.warning(f"Ошибка вычисления центральности для {table}: {e}")

            importance_scores[table] = score

        return importance_scores

    def prioritize_analysis_plan(self,
                                 base_plan: List[str],
                                 schema_info: Dict,
                                 domain_context: DomainContext,
                                 feedback_system: AdaptiveFeedbackSystem) -> List[str]:
        """Переупорядочивает план анализа по приоритетам"""

        # Вычисляем важность таблиц
        table_importance = self.calculate_table_importance(schema_info)

        # Получаем предпочтительные типы вопросов
        preferred_types = feedback_system.get_preferred_question_types()

        # Создаем приоритетные вопросы
        prioritized_questions = []

        # 1. Высокоприоритетные вопросы для важных таблиц
        important_tables = sorted(table_importance.items(), key=lambda x: x[1], reverse=True)[:3]

        for table, importance in important_tables:
            if table in schema_info and schema_info[table].get('row_count', 0) > 0:
                prioritized_questions.append(
                    f"Приоритетный анализ ключевой таблицы '{table}' (важность: {importance:.2f})")

        # 2. Вопросы, специфичные для домена
        if domain_context.domain_type != 'general':
            domain_questions = self._get_domain_priority_questions(domain_context)
            prioritized_questions.extend(domain_questions[:2])

        # 3. Вопросы предпочтительных типов
        type_questions = self._get_questions_by_type(base_plan, preferred_types)
        prioritized_questions.extend(type_questions)

        # 4. Оставшиеся вопросы из базового плана
        remaining_questions = [q for q in base_plan if q not in prioritized_questions]
        prioritized_questions.extend(remaining_questions)

        logger.info(f"Приоритизация завершена: {len(prioritized_questions)} вопросов")
        return prioritized_questions

    def _get_domain_priority_questions(self, domain_context: DomainContext) -> List[str]:
        """Получает приоритетные вопросы для домена"""

        questions = []

        if domain_context.domain_type == 'ecommerce':
            questions = [
                "Анализ ключевых метрик продаж и прибыльности",
                "Исследование поведения самых ценных клиентов"
            ]
        elif domain_context.domain_type == 'crm':
            questions = [
                "Анализ эффективности воронки продаж",
                "Исследование факторов успешного закрытия сделок"
            ]
        elif domain_context.domain_type == 'analytics':
            questions = [
                "Анализ пользовательского поведения и конверсий",
                "Исследование источников трафика и их эффективности"
            ]

        return questions

    def _get_questions_by_type(self, base_plan: List[str], preferred_types: List[str]) -> List[str]:
        """Фильтрует вопросы по предпочтительным типам"""

        type_questions = []

        for pref_type in preferred_types:
            for question in base_plan:
                question_type = self._categorize_question(question)
                if question_type == pref_type and question not in type_questions:
                    type_questions.append(question)

        return type_questions

    def _categorize_question(self, question: str) -> str:
        """Категоризирует вопрос по типу"""

        question_lower = question.lower()

        if any(keyword in question_lower for keyword in ['тренд', 'динамика', 'время', 'дата']):
            return 'temporal'
        elif any(keyword in question_lower for keyword in ['корреляция', 'связь', 'зависимость']):
            return 'correlation'
        elif any(keyword in question_lower for keyword in ['аномалия', 'выброс', 'отклонение']):
            return 'anomaly'
        elif any(keyword in question_lower for keyword in ['группировка', 'сегментация', 'категория']):
            return 'segmentation'
        else:
            return 'general'


# ================== ОБНОВЛЕННЫЕ АГЕНТЫ ==================

class BaseAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.client = openai_client
        self.model = model


class EnhancedOrchestrator(BaseAgent):
    """Улучшенный оркестратор с ML и адаптивностью"""

    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.schema_cache = None
        self.processed_questions = set()

        # Инициализация новых компонентов
        self.ml_detector = MLPatternDetector()
        self.domain_analyzer = DomainAnalyzer()
        self.feedback_system = AdaptiveFeedbackSystem()
        self.validator = AdvancedValidator(engine)
        self.prioritizer = IntelligentPrioritizer(engine)

        # Кэш для обнаруженных паттернов
        self.detected_patterns = []

        self.inspector = inspect(engine)
        self.table_names = [name for name in self.inspector.get_table_names() if name != 'alembic_version']

    def get_comprehensive_schema(self) -> Dict[str, Any]:
        """Получает полную информацию о схеме с ML-анализом"""

        if self.schema_cache:
            return self.schema_cache

        inspector = inspect(self.engine)
        schema_info = {}

        for table_name in inspector.get_table_names():
            if table_name == 'alembic_version':
                continue

            try:
                columns = inspector.get_columns(table_name)

                with self.engine.connect() as conn:
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.scalar()

                    if row_count > 0:
                        # Получаем данные для ML-анализа
                        sample_result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 1000"))
                        sample_data = sample_result.fetchall()

                        # Создаем DataFrame для анализа
                        if sample_data:
                            column_names = [col['name'] for col in columns]
                            df = pd.DataFrame(sample_data, columns=column_names)

                            # Обнаруживаем паттерны с помощью ML
                            patterns = self.ml_detector.detect_all_patterns(df, table_name)
                            self.detected_patterns.extend(patterns)

                        # Статистика по колонкам
                        column_stats = {}
                        for col in columns:
                            col_name = col['name']
                            col_type = str(col['type'])

                            if 'varchar' in col_type.lower() or 'text' in col_type.lower():
                                try:
                                    unique_result = conn.execute(text(
                                        f"SELECT COUNT(DISTINCT {col_name}) as unique_count FROM {table_name}"
                                    ))
                                    unique_count = unique_result.scalar()

                                    if unique_count <= 50:
                                        examples_result = conn.execute(text(
                                            f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 10"
                                        ))
                                        examples = [row[0] for row in examples_result.fetchall()]
                                        column_stats[col_name] = {
                                            'type': col_type,
                                            'unique_count': unique_count,
                                            'examples': examples
                                        }
                                    else:
                                        column_stats[col_name] = {
                                            'type': col_type,
                                            'unique_count': unique_count
                                        }
                                except Exception as e:
                                    column_stats[col_name] = {'type': col_type, 'error': str(e)}
                            else:
                                column_stats[col_name] = {'type': col_type}

                        schema_info[table_name] = {
                            'row_count': row_count,
                            'columns': column_stats,
                            'sample_data': sample_data[:5],  # Ограничиваем размер
                            'ml_patterns': len([p for p in self.detected_patterns if table_name in p.tables_involved])
                        }
                    else:
                        schema_info[table_name] = {'row_count': 0, 'columns': {}, 'sample_data': []}

            except Exception as e:
                logger.error(f"Ошибка анализа таблицы {table_name}: {e}")
                schema_info[table_name] = {'row_count': 0, 'columns': {}, 'sample_data': []}

        self.schema_cache = schema_info
        return schema_info

    def create_intelligent_plan(self) -> List[str]:
        """Создает интеллектуальный план с использованием ML и контекста домена"""

        logger.info("Создание интеллектуального плана анализа...")

        schema = self.get_comprehensive_schema()
        populated_tables = {k: v for k, v in schema.items() if v.get('row_count', 0) > 0}

        if not populated_tables:
            return ["Проверить наличие данных в базе"]

        # Определяем контекст домена
        domain_context = self.domain_analyzer.detect_domain(list(populated_tables.keys()), schema)
        logger.info(f"Обнаружен домен: {domain_context.domain_type} (уверенность: {domain_context.confidence:.2f})")

        # Создаем базовый план
        base_plan = []

        # Добавляем обзорные вопросы
        base_plan.append("Общий обзор структуры базы данных и основных метрик")

        # Добавляем вопросы для каждой таблицы
        for table_name in populated_tables.keys():
            base_plan.append(f"Детальный анализ таблицы '{table_name}' с поиском паттернов")

        # Добавляем вопросы на основе ML-паттернов
        for pattern in self.detected_patterns:
            if pattern.confidence > 0.7:
                base_plan.append(f"Исследование обнаруженного паттерна: {pattern.description}")

        # Добавляем вопросы для домена
        domain_questions = self.domain_analyzer.generate_domain_questions(domain_context, schema)
        base_plan.extend(domain_questions)

        # Применяем интеллектуальную приоритизацию
        prioritized_plan = self.prioritizer.prioritize_analysis_plan(
            base_plan, schema, domain_context, self.feedback_system
        )

        logger.info(f"Создан интеллектуальный план из {len(prioritized_plan)} вопросов")
        return prioritized_plan

    def process_evaluation_with_feedback(self, evaluation: dict, session_memory: list,
                                         analysis_plan: list, current_question: str,
                                         user_rating: int = None, user_feedback: str = ""):
        """Обрабатывает результаты с учетом обратной связи"""

        self.processed_questions.add(current_question)

        if evaluation.get('finding'):
            finding = evaluation['finding']
            session_memory.append(finding)

            # Собираем обратную связь если она есть
            if user_rating is not None:
                self.feedback_system.collect_feedback(
                    current_question, finding, user_rating, user_feedback
                )

        # Генерируем адаптивные гипотезы
        new_hypotheses = evaluation.get('new_hypotheses', [])
        if new_hypotheses:
            # Адаптируем стратегию на основе обратной связи
            strategy = self.feedback_system.adapt_analysis_strategy()

            # Фильтруем гипотезы по предпочтительным типам
            filtered_hypotheses = self._filter_hypotheses_by_strategy(new_hypotheses, strategy)

            unique_hypotheses = [h for h in filtered_hypotheses if h not in self.processed_questions]
            if unique_hypotheses:
                limited_hypotheses = unique_hypotheses[:2]
                analysis_plan.extend(limited_hypotheses)
                logger.info(f"Добавлено {len(limited_hypotheses)} адаптивных гипотез")

    def _filter_hypotheses_by_strategy(self, hypotheses: List[str], strategy: Dict[str, Any]) -> List[str]:
        """Фильтрует гипотезы на основе стратегии обратной связи"""

        filtered = []
        preferred_types = strategy.get('preferred_question_types', [])

        for hypothesis in hypotheses:
            hypothesis_type = self.prioritizer._categorize_question(hypothesis)

            # Приоритизируем предпочтительные типы
            if hypothesis_type in preferred_types[:2]:  # Топ-2 предпочтительных типа
                filtered.append(hypothesis)
            elif len(filtered) < 3:  # Добавляем остальные если места есть
                filtered.append(hypothesis)

        return filtered

    def get_ml_insights_summary(self) -> Dict[str, Any]:
        """Возвращает сводку ML-инсайтов в сериализуемом формате"""
        pattern_summary = {}

        for pattern in self.detected_patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in pattern_summary:
                pattern_summary[pattern_type] = []

            # Используем метод to_dict() для сериализации
            pattern_summary[pattern_type].append({
                'description': pattern.description,
                'confidence': float(pattern.confidence),
                'tables': pattern.tables_involved
            })

        return {
            'total_patterns': len(self.detected_patterns),
            'pattern_types': pattern_summary,
            'high_confidence_patterns': [
                p.to_dict() for p in self.detected_patterns if p.confidence > 0.8
            ]
        }

    def get_analysis_diversity_report(self, session_memory: list) -> dict:
        """Создает отчет о разнообразии проведенного анализа."""
        analyzed_tables = self._get_analyzed_tables(session_memory)

        return {
            'total_tables': len(self.table_names),
            'analyzed_tables': len(analyzed_tables),
            'coverage_percentage': len(analyzed_tables) / len(self.table_names) * 100 if self.table_names else 0,
            'table_analysis_distribution': analyzed_tables,
            'underanalyzed_tables': [t for t in self.table_names if analyzed_tables.get(t, 0) < 2]
        }

    def _get_analyzed_tables(self, session_memory: list) -> dict:
        """Подсчитывает количество анализов по каждой таблице."""
        table_count = {}

        for finding in session_memory:
            analyzed_tables = finding.get('analyzed_tables', [])
            for table in analyzed_tables:
                table_count[table] = table_count.get(table, 0) + 1

        return table_count


class EnhancedSQLCoder(BaseAgent):
    """Улучшенный SQL-кодер с валидацией"""

    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.inspector = inspect(engine)
        self.table_names = [name for name in self.inspector.get_table_names() if name != 'alembic_version']
        self.validator = AdvancedValidator(engine)

        self.db = SQLDatabase(
            engine=engine,
            include_tables=self.table_names,
            sample_rows_in_table_info=5
        )

        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0.1,
            api_key=config.API_KEY
        )

    def run_with_validation(self, question: str) -> dict:
        """Выполняет анализ с продвинутой валидацией"""

        logger.info(f"Анализ с валидацией: '{question}'")

        table_info = self.db.get_table_info()

        system_prompt = f"""Вы - эксперт по анализу данных с фокусом на качество и безопасность.

        **КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ:**
        1. ВСЕГДА используйте LIMIT для ограничения результатов
        2. НИКОГДА не используйте операции изменения данных (INSERT, UPDATE, DELETE, DROP)
        3. Проверяйте корректность названий таблиц и колонок
        4. Используйте агрегации для получения значимых инсайтов
        5. Обрабатывайте NULL значения корректно

        **СХЕМА БАЗЫ ДАННЫХ:**
        {table_info}

        **СТРАТЕГИЯ АНАЛИЗА:**
        - Начинайте с простых запросов
        - Используйте подзапросы для сложной логики
        - Применяйте оконные функции для аналитики
        - Группируйте данные для выявления паттернов

        Создайте безопасный и эффективный SQL запрос.
        """

        df = pd.DataFrame()
        sql_query = None
        error_msg = None
        validation_result = None

        try:
            # Используем агента для генерации SQL
            agent_executor = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_type="openai-tools",
                verbose=True,
                agent_kwargs={"system_message": system_prompt},
                handle_parsing_errors=True,
                max_iterations=3
            )

            agent_result = agent_executor.invoke({"input": question})
            sql_query = self.extract_sql_from_agent_response(agent_result)

            # Валидируем извлеченный SQL
            if sql_query:
                is_valid, validation_msg = self.validator.validate_sql_query(sql_query)
                validation_result = {'is_valid': is_valid, 'message': validation_msg}

                if is_valid:
                    df = self.execute_query_safely(sql_query)

                    # Валидируем качество данных
                    if not df.empty:
                        data_quality = self.validator.validate_data_quality(df, "result")
                        validation_result['data_quality'] = data_quality
                else:
                    error_msg = f"Валидация не пройдена: {validation_msg}"
                    logger.error(error_msg)

        except Exception as e:
            logger.error(f"Ошибка выполнения с валидацией: {e}")
            error_msg = str(e)

        # Fallback стратегия если основной запрос не прошел валидацию
        if df.empty and validation_result and not validation_result.get('is_valid'):
            logger.warning("Используем fallback стратегию после неудачной валидации")
            fallback_result = self.run_fallback_strategy(question)
            df = fallback_result.get('data', pd.DataFrame())
            if not df.empty:
                sql_query = fallback_result.get('sql_query')
                validation_result = {'is_valid': True, 'message': 'Fallback запрос'}

        return {
            "question": question,
            "data": df,
            "sql_query": sql_query,
            "error": error_msg,
            "validation": validation_result,
            "row_count": len(df),
            "column_count": len(df.columns) if not df.empty else 0
        }

    def run_fallback_strategy(self, question: str) -> dict:
        """Запускает безопасную fallback стратегию"""

        # Создаем простые, гарантированно валидные запросы
        safe_queries = []

        for table in self.table_names:
            safe_queries.extend([
                f"SELECT COUNT(*) as row_count FROM {table}",
                f"SELECT * FROM {table} LIMIT 5"
            ])

        # Пытаемся выполнить безопасные запросы
        for query in safe_queries:
            try:
                is_valid, _ = self.validator.validate_sql_query(query)
                if is_valid:
                    df = self.execute_query_safely(query)
                    if not df.empty:
                        return {
                            "data": df,
                            "sql_query": query,
                            "fallback_used": True
                        }
            except Exception as e:
                logger.error(f"Fallback запрос не удался: {e}")
                continue

        return {"data": pd.DataFrame(), "sql_query": None, "fallback_used": True}

    def execute_query_safely(self, query: str) -> pd.DataFrame:
        """Безопасно выполняет SQL запрос"""

        try:
            # Добавляем LIMIT если его нет
            if 'limit' not in query.lower() and 'count(' not in query.lower():
                query = f"{query.rstrip(';')} LIMIT 100"

            with self.engine.connect() as connection:
                logger.info(f"Выполнение запроса: {query[:100]}...")
                df = pd.read_sql(query, connection)
                logger.info(f"Получено {len(df)} строк, {len(df.columns)} колонок")
                return df

        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            return pd.DataFrame()

    def extract_sql_from_agent_response(self, agent_result: dict) -> str:
        """Извлекает SQL запрос из ответа агента"""

        try:
            # Проверяем intermediate_steps
            if 'intermediate_steps' in agent_result:
                for step in agent_result['intermediate_steps']:
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, observation = step
                        if hasattr(action, 'tool_input'):
                            tool_input = action.tool_input
                            if isinstance(tool_input, dict) and 'query' in tool_input:
                                return tool_input['query']

            # Пытаемся найти SQL в тексте ответа
            output_text = agent_result.get('output', '')
            sql_patterns = [
                r'``````',
                r'``````',
                r'Query:\s*(SELECT.*?)(?:\n|$)',
                r'(SELECT.*?)(?:\n|$)'
            ]

            for pattern in sql_patterns:
                match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(1).strip()

            return None

        except Exception as e:
            logger.error(f"Ошибка извлечения SQL: {e}")
            return None


# Функция создания визуализации (улучшенная версия)
def create_enhanced_visualization(df: pd.DataFrame, question: str, ml_patterns: List[DataPattern] = None) -> str:
    """Создает улучшенную визуализацию с учетом ML-паттернов"""

    if df.empty:
        return None

    try:
        fig = None

        # Если есть обнаруженные паттерны, используем их для визуализации
        if ml_patterns:
            for pattern in ml_patterns:
                if pattern.pattern_type == 'anomaly' and pattern.confidence > 0.7:
                    # Создаем визуализацию аномалий
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 2:
                        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                                         title=f"Аномалии: {question}",
                                         color_discrete_sequence=[
                                             'red' if i in pattern.metadata.get('anomaly_indices', []) else 'blue'
                                             for i in range(len(df))])
                        break

        # Стандартная логика если нет специальных паттернов
        if fig is None:
            if len(df.columns) == 1:
                col = df.columns[0]
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig = px.histogram(df, x=col, title=f"Распределение: {question}")
                else:
                    value_counts = df[col].value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                                 title=f"Топ значения: {question}")

            elif len(df.columns) == 2:
                x_col, y_col = df.columns[0], df.columns[1]
                if pd.api.types.is_numeric_dtype(df[y_col]):
                    fig = px.bar(df.head(20), x=x_col, y=y_col, title=question)
                else:
                    fig = px.bar(df.head(20), x=x_col, y=y_col, title=question)

            else:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) >= 2:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, title=f"Корреляция: {question}",
                                    color_continuous_scale='RdBu_r')

        if fig:
            fig.update_layout(
                title_font_size=16,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                height=500,
                margin=dict(l=50, r=50, t=80, b=50)
            )

            img_bytes = io.BytesIO()
            fig.write_image(img_bytes, format="png", scale=2, width=800, height=500)
            img_bytes.seek(0)

            file_name = f"charts/{uuid.uuid4()}.png"
            s3_client.upload_fileobj(
                img_bytes,
                config.S3_BUCKET_NAME,
                file_name,
                ExtraArgs={'ContentType': 'image/png', 'ACL': 'public-read'}
            )

            return f"https://{config.S3_BUCKET_NAME}.s3.{config.AWS_DEFAULT_REGION}.amazonaws.com/{file_name}"

    except Exception as e:
        logger.error(f"Ошибка создания визуализации: {e}")

    return None


# Обновленный Critic с ML-анализом
class EnhancedCritic(BaseAgent):
    """Улучшенный критик с ML-анализом и контекстным пониманием"""

    def __init__(self, ml_detector: MLPatternDetector = None, **kwargs):
        super().__init__(**kwargs)
        self.ml_detector = ml_detector or MLPatternDetector()

    def evaluate_with_ml(self, execution_result: dict) -> dict:
        """Оценивает результаты с использованием ML-анализа"""

        df = execution_result.get('data', pd.DataFrame())
        question = execution_result.get('question', 'N/A')
        sql_query = execution_result.get('sql_query', 'N/A')
        validation = execution_result.get('validation', {})

        # Обработка ошибок валидации
        if validation and not validation.get('is_valid', True):
            return {
                "is_success": False,
                "finding": {
                    "question": question,
                    "summary": f"Ошибка валидации: {validation.get('message', 'Неизвестная ошибка')}",
                    "sql_query": sql_query,
                    "validation_issues": validation
                },
                "new_hypotheses": []
            }

        # Обработка пустых результатов
        if df.empty:
            return {
                "is_success": True,
                "finding": {
                    "question": question,
                    "summary": "Запрос выполнен успешно, но данные отсутствуют",
                    "sql_query": sql_query,
                    "chart_url": None,
                    "data_preview": None
                },
                "new_hypotheses": [
                    "Проверить условия фильтрации данных",
                    "Исследовать общую статистику по таблицам"
                ]
            }

        try:
            # ML-анализ данных
            table_name = self._extract_table_name(sql_query)
            ml_patterns = self.ml_detector.detect_all_patterns(df, table_name)

            # Создаем визуализацию с учетом ML-паттернов
            chart_url = create_enhanced_visualization(df, question, ml_patterns)

            # Анализируем данные
            data_analysis = self._analyze_dataframe_with_ml(df, ml_patterns)

            # Превью данных
            data_preview = df.head(10).to_dict('records')

            # Генерируем инсайты с учетом ML-паттернов
            insights = self._generate_ml_insights(df, question, data_analysis, ml_patterns)

            # Генерируем гипотезы на основе ML-паттернов
            new_hypotheses = self._generate_ml_hypotheses(df, question, ml_patterns)

            return {
                "is_success": True,
                "finding": {
                    "question": question,
                    "summary": insights,
                    "sql_query": sql_query,
                    "chart_url": chart_url,
                    "data_preview": data_preview,
                    "data_stats": data_analysis,
                    "ml_patterns": [
                        p.to_dict() for p in ml_patterns  # Используем to_dict()
                    ],
                    "validation": validation
                },
                "new_hypotheses": new_hypotheses
            }

        except Exception as e:
            logger.error(f"Ошибка ML-анализа: {e}")
            return {
                "is_success": False,
                "finding": {
                    "question": question,
                    "summary": f"Ошибка анализа: {e}",
                    "sql_query": sql_query
                },
                "new_hypotheses": []
            }

    def _extract_table_name(self, sql_query: str) -> str:
        """Извлекает название основной таблицы из SQL запроса"""

        if not sql_query:
            return "unknown"

        # Простое извлечение из FROM clause
        match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
        if match:
            return match.group(1)

        return "unknown"

    def _analyze_dataframe_with_ml(self, df: pd.DataFrame, ml_patterns: List[DataPattern]) -> dict:
        """Анализирует DataFrame с учетом ML-паттернов"""

        analysis = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_stats": {},
            "categorical_stats": {},
            "ml_insights": {}
        }

        # Базовая статистика
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            analysis["numeric_stats"][col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            analysis["categorical_stats"][col] = {
                "unique_count": len(value_counts),
                "top_values": value_counts.head(5).to_dict()
            }

        # ML-инсайты
        for pattern in ml_patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in analysis["ml_insights"]:
                analysis["ml_insights"][pattern_type] = []

            analysis["ml_insights"][pattern_type].append({
                "description": pattern.description,
                "confidence": pattern.confidence,
                "metadata": pattern.metadata
            })

        return analysis

    def _generate_ml_insights(self, df: pd.DataFrame, question: str,
                              data_analysis: dict, ml_patterns: List[DataPattern]) -> str:
        """Генерирует инсайты с учетом ML-паттернов"""

        try:
            # Подготавливаем информацию о ML-паттернах
            ml_summary = ""
            if ml_patterns:
                high_confidence_patterns = [p for p in ml_patterns if p.confidence > 0.7]
                if high_confidence_patterns:
                    ml_summary = f"Обнаружено {len(high_confidence_patterns)} значимых паттернов:\n"
                    for pattern in high_confidence_patterns:
                        ml_summary += f"- {pattern.description} (уверенность: {pattern.confidence:.2f})\n"

            prompt = f"""Создайте комплексный анализ данных с учетом ML-паттернов.

            Исходный вопрос: "{question}"

            Базовая статистика:
            - Размер данных: {data_analysis['shape']}
            - Числовые метрики: {data_analysis['numeric_stats']}
            - Категориальные данные: {data_analysis['categorical_stats']}

            ML-паттерны:
            {ml_summary}

            Создайте краткий (3-4 предложения) экспертный анализ, который:
            1. Отвечает на исходный вопрос
            2. Объясняет обнаруженные ML-паттерны
            3. Предоставляет практические рекомендации
            4. Указывает на потенциальные бизнес-выводы

            Ответ на русском языке, профессиональным тоном.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Ошибка генерации ML-инсайтов: {e}")
            base_insight = f"Анализ {len(df)} записей выявил {len(ml_patterns)} паттернов."
            if ml_patterns:
                base_insight += f" Наиболее значимый: {ml_patterns[0].description}"
            return base_insight

    def _generate_ml_hypotheses(self, df: pd.DataFrame, question: str,
                                ml_patterns: List[DataPattern]) -> List[str]:
        """Генерирует гипотезы на основе ML-паттернов"""

        hypotheses = []

        for pattern in ml_patterns:
            if pattern.confidence > 0.7:
                if pattern.pattern_type == 'anomaly':
                    hypotheses.append(f"Детальное исследование аномалий в {', '.join(pattern.tables_involved)}")
                    hypotheses.append(f"Поиск причин аномального поведения в данных")

                elif pattern.pattern_type == 'clustering':
                    hypotheses.append(f"Анализ характеристик каждого кластера в {', '.join(pattern.tables_involved)}")
                    hypotheses.append(f"Бизнес-интерпретация обнаруженных групп")

                elif pattern.pattern_type == 'correlation':
                    hypotheses.append(f"Углубленный анализ корреляций в {', '.join(pattern.tables_involved)}")
                    hypotheses.append(f"Поиск причинно-следственных связей")

        # Добавляем общие гипотезы на основе типов данных
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            hypotheses.append(f"Временной анализ числовых метрик")

        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            hypotheses.append(f"Анализ трендов по временным данным")

        return hypotheses[:4]  # Ограничиваем количество


# Главная функция для запуска улучшенного анализа
def run_enhanced_analysis(engine, max_questions: int = 15, enable_feedback: bool = True) -> dict:
    """Запускает улучшенный анализ с ML и адаптивностью"""

    logger.info("Запуск улучшенного анализа с ML и адаптивностью")

    # Инициализация улучшенных компонентов
    orchestrator = EnhancedOrchestrator(engine)
    sql_coder = EnhancedSQLCoder(engine)
    critic = EnhancedCritic(orchestrator.ml_detector)

    # Создаем интеллектуальный план
    analysis_plan = orchestrator.create_intelligent_plan()
    session_memory = []

    logger.info(f"Создан интеллектуальный план из {len(analysis_plan)} вопросов")

    # Выполняем анализ
    questions_processed = 0

    for question in analysis_plan:
        if questions_processed >= max_questions:
            break

        logger.info(f"Обработка вопроса {questions_processed + 1}/{max_questions}: {question}")

        # Получаем данные с валидацией
        execution_result = sql_coder.run_with_validation(question)

        # Оцениваем результат с ML-анализом
        evaluation = critic.evaluate_with_ml(execution_result)

        # Обрабатываем результат с адаптацией
        # В реальном применении здесь можно добавить пользовательскую обратную связь
        simulated_rating = 4 if evaluation.get('is_success') else 2

        orchestrator.process_evaluation_with_feedback(
            evaluation, session_memory, analysis_plan, question,
            simulated_rating if enable_feedback else None
        )

        questions_processed += 1

    # Создаем итоговый отчет
    final_report = {
        "executive_summary": f"Проведен интеллектуальный анализ {questions_processed} вопросов с использованием ML",
        "detailed_findings": session_memory,
        "ml_insights": orchestrator.get_ml_insights_summary(),
        "questions_processed": questions_processed,
        "feedback_enabled": enable_feedback
    }

    if enable_feedback:
        final_report["adaptive_strategy"] = orchestrator.feedback_system.adapt_analysis_strategy()

    logger.info("Улучшенный анализ завершен")
    return final_report


def get_database_health_check(engine) -> dict:
    """
    Проверяет, удалось ли подключиться к БД и есть ли в ней данные.
    Возвращает словарь с ключами:
      - connection: bool
      - has_data: bool
      - total_rows: int
    """
    try:
        with engine.connect() as conn:
            inspector = inspect(engine)
            # Игнорируем служебную таблицу alembic_version
            tables = [t for t in inspector.get_table_names() if t != 'alembic_version']
            total_rows = 0
            for table in tables:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0
                total_rows += count

            return {
                "connection": True,
                "has_data": total_rows > 0,
                "total_rows": total_rows
            }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "connection": False,
            "has_data": False,
            "total_rows": 0
        }


# Экспорт основных классов
__all__ = [
    'EnhancedOrchestrator',
    'EnhancedSQLCoder',
    'EnhancedCritic',
    'MLPatternDetector',
    'DomainAnalyzer',
    'AdaptiveFeedbackSystem',
    'AdvancedValidator',
    'IntelligentPrioritizer',
    'run_enhanced_analysis'
]

