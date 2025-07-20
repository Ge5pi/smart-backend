# services/dataframe_analyzer.py - ПОЛНАЯ ВЕРСИЯ с SmartGPT

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from services.dataframe_manager import DataFrameManager
from datetime import datetime
import re
import sys
import os
from services.gpt_analyzer import SmartGPTAnalyzer

# Добавляем путь для импорта utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_serializer import convert_to_serializable, clean_dataframe_for_json

logger = logging.getLogger(__name__)


class DataFrameAnalyzer:
    """Аналитический движок для работы с DataFrame с полной SmartGPT интеграцией"""

    def __init__(self, df_manager: DataFrameManager):
        self.df_manager = df_manager
        self.analysis_cache = {}
        self.gpt_analyzer = SmartGPTAnalyzer()

    def _extract_mentioned_tables(self, question: str) -> List[str]:
        """Извлекает все упомянутые таблицы"""
        mentioned = []
        question_lower = question.lower()

        for table_name in self.df_manager.tables.keys():
            if table_name.lower() in question_lower:
                mentioned.append(table_name)

        # Если ничего не найдено, возвращаем самую большую таблицу
        if not mentioned and self.df_manager.tables:
            largest_table = max(self.df_manager.tables.items(), key=lambda x: len(x[1]))[0]
            mentioned = [largest_table]

        return mentioned

    def _find_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Находит колонки с датами"""
        date_cols = []

        for col in df.columns:
            # Проверяем по названию
            if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
                date_cols.append(col)
            # Проверяем по типу данных
            elif df[col].dtype == 'datetime64[ns]':
                date_cols.append(col)
            # Пытаемся преобразовать к дате первые несколько значений
            else:
                try:
                    sample = df[col].dropna().head(5)
                    if len(sample) > 0:
                        pd.to_datetime(sample.iloc[0], errors='raise')
                        date_cols.append(col)
                except:
                    continue

        return date_cols

    def _categorize_question(self, question: str) -> str:
        """Категоризирует тип вопроса"""
        question_lower = question.lower()

        if any(keyword in question_lower for keyword in ['обзор', 'общий', 'статистика', 'overview', 'структура']):
            return 'overview'
        elif any(keyword in question_lower for keyword in ['таблица', 'table', 'анализ таблицы']):
            return 'table_analysis'
        elif any(keyword in question_lower for keyword in ['связь', 'связи', 'отношения', 'relation']):
            return 'relationship_analysis'
        elif any(keyword in question_lower for keyword in ['бизнес', 'метрики', 'kpi', 'инсайт']):
            return 'business_insights'
        elif any(keyword in question_lower for keyword in ['корреляция', 'зависимость', 'связанность']):
            return 'correlation'
        elif any(keyword in question_lower for keyword in ['тренд', 'динамика', 'время', 'временной']):
            return 'trend_analysis'
        elif any(keyword in question_lower for keyword in ['аномалии', 'выбросы', 'anomaly', 'outlier']):
            return 'anomalies'
        else:
            return 'general'

    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Анализирует вопрос и возвращает результат на основе DataFrame операций"""
        logger.info(f"Анализ вопроса: {question}")

        # Определяем тип анализа
        analysis_type = self._categorize_question(question)

        try:
            if analysis_type == 'overview':
                return self._analyze_overview()
            elif analysis_type == 'table_analysis':
                table_name = self._extract_table_name(question)
                return self._analyze_single_table(table_name)
            elif analysis_type == 'relationship_analysis':
                return self._analyze_relationships()
            elif analysis_type == 'aggregation':
                return self._analyze_aggregations(question)
            elif analysis_type == 'trend_analysis':
                return self._analyze_trends(question)
            elif analysis_type == 'correlation':
                return self._analyze_correlations(question)
            elif analysis_type == 'comparison':
                return self._analyze_comparison(question)
            elif analysis_type == 'anomalies':
                return self._analyze_anomalies(question)
            elif analysis_type == 'business_insights':
                return self._analyze_business_metrics(question)
            elif analysis_type == 'data_quality':
                return self._analyze_data_quality(question)
            elif analysis_type == 'statistical_insights':
                return self._analyze_statistical_insights(question)
            elif analysis_type == 'predictive_analysis':
                return self._analyze_predictive_patterns(question)
            else:
                return self._analyze_general(question)

        except Exception as e:
            logger.error(f"Ошибка анализа вопроса '{question}': {e}")
            return {
                'question': question,
                'error': str(e),
                'data': [],
                'summary': f'Не удалось проанализировать: {str(e)}',
                'analyzed_tables': []
            }

    def _extract_table_name(self, question: str) -> str:
        """Извлекает название таблицы из вопроса"""
        question_lower = question.lower()

        # Прямое совпадение
        for table_name in self.df_manager.tables.keys():
            if table_name.lower() in question_lower:
                return table_name

        # Берем самую большую таблицу по умолчанию
        if self.df_manager.tables:
            return max(self.df_manager.tables.items(), key=lambda x: len(x[1]))[0]

        return "unknown"

    def _analyze_overview(self) -> Dict[str, Any]:
        """Общий обзор данных с GPT анализом"""
        overview_data = []

        for table_name, df in self.df_manager.tables.items():
            # Базовая статистика
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            datetime_cols = len(df.select_dtypes(include=['datetime64']).columns)
            null_count = df.isnull().sum().sum()

            # Дополнительная статистика
            duplicate_rows = df.duplicated().sum()
            unique_rows = len(df.drop_duplicates())

            overview_data.append({
                'table_name': table_name,
                'row_count': int(len(df)),
                'column_count': int(len(df.columns)),
                'numeric_columns': int(numeric_cols),
                'categorical_columns': int(categorical_cols),
                'datetime_columns': int(datetime_cols),
                'null_values': int(null_count),
                'null_percentage': round(float((null_count / (len(df) * len(df.columns))) * 100), 2),
                'duplicate_rows': int(duplicate_rows),
                'unique_rows': int(unique_rows),
                'data_completeness': round(float((1 - null_count / (len(df) * len(df.columns))) * 100), 2),
                'memory_usage_mb': round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 2)
            })

        overview_df = pd.DataFrame(overview_data)

        # Информация о связях
        relations_info = []
        for rel in self.df_manager.relations:
            relations_info.append({
                'from_table': rel.from_table,
                'to_table': rel.to_table,
                'from_column': rel.from_column,
                'to_column': rel.to_column,
                'relation_type': rel.relation_type
            })

        total_rows = overview_df['row_count'].sum() if not overview_df.empty else 0
        total_tables = len(self.df_manager.tables)
        relations_count = len(self.df_manager.relations)
        total_memory = overview_df['memory_usage_mb'].sum() if not overview_df.empty else 0
        avg_completeness = overview_df['data_completeness'].mean() if not overview_df.empty else 0

        # GPT анализ обзора
        gpt_context = {
            'total_tables': total_tables,
            'total_rows': total_rows,
            'relations_count': relations_count,
            'avg_completeness': avg_completeness,
            'overview_data': overview_data[:3]  # Первые 3 таблицы для контекста
        }

        try:
            gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                df=overview_df,
                table_name='database_overview',
                analysis_type='business_insights',
                context=gpt_context
            )
            gpt_insights = gpt_analysis.get('gpt_analysis', '')
        except Exception as e:
            logger.error(f"Ошибка GPT анализа обзора: {e}")
            gpt_insights = 'GPT анализ обзора недоступен'

        summary = (f"🏠 **База данных содержит {total_tables} таблиц с {total_rows:,} записями**\n\n"
                   f"📊 Обнаружено {relations_count} связей между таблицами\n"
                   f"💾 Общий объем данных в памяти: {total_memory:.2f} MB\n"
                   f"📈 Средняя полнота данных: {avg_completeness:.1f}%\n\n"
                   f"🤖 **GPT Инсайты:**\n{gpt_insights}")

        return {
            'question': 'Умный обзор базы данных',
            'data': clean_dataframe_for_json(overview_df),
            'summary': summary,
            'analyzed_tables': list(self.df_manager.tables.keys()),
            'relations': relations_info,
            'gpt_insights': gpt_insights,
            'overview_stats': {
                'total_tables': total_tables,
                'total_rows': total_rows,
                'relations_count': relations_count,
                'avg_completeness': round(avg_completeness, 1),
                'total_memory_mb': round(total_memory, 2)
            },
            'chart_data': self._prepare_chart_data_safe(overview_df, 'bar', 'table_name', 'row_count')
        }

    def _create_table_not_found_result(self, table_name: str) -> Dict[str, Any]:
        """Создает результат для случая, когда таблица не найдена"""
        available_tables = list(self.df_manager.tables.keys())

        return {
            'question': f'Анализ таблицы {table_name}',
            'error': f'Таблица {table_name} не найдена',
            'data': [],
            'summary': f'Таблица {table_name} не найдена. Доступные таблицы: {", ".join(available_tables)}',
            'analyzed_tables': [],
            'available_tables': available_tables
        }


    def _analyze_single_table(self, table_name: str) -> Dict[str, Any]:
        """Детальный анализ одной таблицы с полным GPT анализом"""
        if table_name not in self.df_manager.tables:
            # Пытаемся найти таблицу по частичному совпадению
            matching_tables = [t for t in self.df_manager.tables.keys()
                               if table_name.lower() in t.lower()]
            if matching_tables:
                table_name = matching_tables[0]
                logger.info(f"Найдена таблица по частичному совпадению: {table_name}")
            else:
                return self._create_table_not_found_result(table_name)

        df = self.df_manager.tables[table_name]

        # Базовая информация
        basic_info = {
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'memory_mb': round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 3),
            'columns_list': list(df.columns)
        }

        # Статистический анализ числовых колонок
        numeric_stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            desc_stats = df[numeric_cols].describe()
            numeric_stats = convert_to_serializable(desc_stats.to_dict())

        # Анализ категориальных колонок
        categorical_stats = self._analyze_categorical_columns(df)

        # Поиск аномалий
        anomalies = self._detect_anomalies_dataframe(df)

        # Анализ пропущенных значений
        missing_analysis = self._analyze_missing_values(df)

        # Корреляционный анализ
        correlations = self._analyze_correlations_single_table(df, table_name)

        # КОМПЛЕКСНЫЙ GPT АНАЛИЗ
        logger.info(f"Запуск комплексного GPT анализа для таблицы {table_name}")

        # Бизнес-инсайты
        try:
            gpt_business_insights = self.gpt_analyzer.analyze_data_with_gpt(
                df=df,
                table_name=table_name,
                analysis_type="business_insights",
                context={
                    'basic_info': basic_info,
                    'anomalies': anomalies,
                    'correlations': correlations,
                    'missing_analysis': missing_analysis
                }
            )
        except Exception as e:
            logger.error(f"Ошибка GPT бизнес-анализа: {e}")
            gpt_business_insights = {'gpt_analysis': 'Бизнес-анализ недоступен'}

        # Анализ качества данных
        try:
            gpt_data_quality = self.gpt_analyzer.analyze_data_with_gpt(
                df=df,
                table_name=table_name,
                analysis_type="data_quality",
                context={
                    'missing_analysis': missing_analysis,
                    'anomalies': anomalies,
                    'categorical_stats': categorical_stats
                }
            )
        except Exception as e:
            logger.error(f"Ошибка GPT анализа качества: {e}")
            gpt_data_quality = {'gpt_analysis': 'Анализ качества недоступен'}

        # Статистические инсайты
        try:
            gpt_statistical = self.gpt_analyzer.analyze_data_with_gpt(
                df=df,
                table_name=table_name,
                analysis_type="statistical_insights",
                context={
                    'numeric_stats': numeric_stats,
                    'correlations': correlations
                }
            ) if len(numeric_cols) > 0 else {'gpt_analysis': ''}
        except Exception as e:
            logger.error(f"Ошибка GPT статистического анализа: {e}")
            gpt_statistical = {'gpt_analysis': ''}

        # GPT анализ корреляций
        correlation_insights = ""
        if correlations:
            try:
                correlation_insights = self.gpt_analyzer.analyze_correlations_with_context(
                    correlations, df, table_name
                )
            except Exception as e:
                logger.error(f"Ошибка GPT анализа корреляций: {e}")
                correlation_insights = 'Анализ корреляций недоступен'

        # Создаем обогащенную сводку
        summary = f"🎯 **Детальный анализ таблицы '{table_name}'**\n\n"
        summary += f"📊 **Базовая информация:** {len(df):,} записей, {len(df.columns)} колонок\n"

        if anomalies:
            summary += f"⚠️ Обнаружено {len(anomalies)} типов аномалий\n"
        if correlations:
            summary += f"🔗 Найдено {len(correlations)} значимых корреляций\n"
        if missing_analysis['total_missing'] > 0:
            summary += f"❌ Пропущенных значений: {missing_analysis['total_missing']} ({missing_analysis['missing_percentage']:.1f}%)\n"

        summary += f"\n💼 **Бизнес-инсайты:**\n{gpt_business_insights.get('gpt_analysis', 'Недоступно')}\n"

        if gpt_data_quality.get('gpt_analysis'):
            summary += f"\n🔍 **Качество данных:**\n{gpt_data_quality.get('gpt_analysis')}\n"

        if gpt_statistical.get('gpt_analysis'):
            summary += f"\n📈 **Статистические находки:**\n{gpt_statistical.get('gpt_analysis')}\n"

        if correlation_insights:
            summary += f"\n🔗 **Анализ корреляций:**\n{correlation_insights}"

        return {
            'question': f'Детальный анализ таблицы {table_name}',
            'data': clean_dataframe_for_json(df.head(10)),
            'summary': summary,
            'analyzed_tables': [table_name],

            # GPT результаты
            'gpt_business_insights': gpt_business_insights.get('gpt_analysis', ''),
            'gpt_data_quality': gpt_data_quality.get('gpt_analysis', ''),
            'gpt_statistical': gpt_statistical.get('gpt_analysis', ''),
            'correlation_insights': correlation_insights,

            # Технические данные
            'basic_info': basic_info,
            'numeric_stats': numeric_stats,
            'categorical_stats': categorical_stats,
            'anomalies': anomalies,
            'correlations': correlations,
            'missing_analysis': missing_analysis,

            'chart_data': self._prepare_chart_data_safe(df, 'histogram', df.columns[0], None)
        }

    def _analyze_business_metrics(self, question: str) -> Dict[str, Any]:
        """Анализ бизнес-метрик с GPT"""
        tables_mentioned = self._extract_mentioned_tables(question)
        if not tables_mentioned:
            tables_mentioned = [max(self.df_manager.tables.items(), key=lambda x: len(x[1]))[0]]

        main_table = tables_mentioned[0]
        df = self.df_manager.tables[main_table]

        # Вычисляем основные бизнес-метрики
        business_metrics = self._calculate_business_metrics(df)

        # Расширенный бизнес-анализ
        advanced_metrics = self._calculate_advanced_business_metrics(df, main_table)
        business_metrics.update(advanced_metrics)

        # GPT анализ бизнес-метрик
        try:
            gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                df=df,
                table_name=main_table,
                analysis_type="business_insights",
                context={
                    "metrics": business_metrics,
                    "question": question,
                    "focus": "business_performance"
                }
            )
            gpt_insights = gpt_analysis.get('gpt_analysis', 'GPT анализ недоступен')
        except Exception as e:
            logger.error(f"Ошибка GPT бизнес-анализа: {e}")
            gpt_insights = 'GPT анализ бизнес-метрик недоступен'

        # Создаем результат
        metrics_df = pd.DataFrame([business_metrics])

        summary = f"💼 **Бизнес-анализ таблицы '{main_table}'**\n\n"
        summary += f"📊 **Ключевые метрики:**\n"
        summary += f"• Общее количество записей: {business_metrics.get('total_records', 0):,}\n"
        summary += f"• Полнота данных: {business_metrics.get('data_completeness', 0):.1f}%\n"
        summary += f"• Уровень дубликатов: {business_metrics.get('duplicate_rate', 0):.1f}%\n"

        if business_metrics.get('revenue_metrics'):
            summary += f"• Финансовые показатели найдены\n"

        summary += f"\n🤖 **GPT Инсайты:**\n{gpt_insights}"

        return {
            'question': question,
            'data': clean_dataframe_for_json(metrics_df),
            'summary': summary,
            'analyzed_tables': [main_table],
            'gpt_insights': gpt_insights,
            'business_metrics': business_metrics,
            'chart_data': self._prepare_business_metrics_chart(business_metrics)
        }

    def _calculate_business_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Вычисляет ключевые бизнес-метрики"""
        metrics = {
            'total_records': len(df),
            'data_completeness': round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1),
            'duplicate_rate': round((df.duplicated().sum() / len(df)) * 100, 1),
            'unique_entities': {},
            'numeric_summaries': {},
            'revenue_metrics': {},
            'customer_metrics': {}
        }

        # Анализ числовых колонок
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            try:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    metrics['numeric_summaries'][col] = {
                        'total': float(col_data.sum()),
                        'average': round(float(col_data.mean()), 2),
                        'median': round(float(col_data.median()), 2),
                        'min': float(col_data.min()),
                        'max': float(col_data.max())
                    }

                    # Специальные метрики для финансовых данных
                    if any(keyword in col_lower for keyword in ['revenue', 'sales', 'amount', 'price', 'cost']):
                        metrics['revenue_metrics'][col] = {
                            'total_revenue': float(col_data.sum()),
                            'avg_transaction': round(float(col_data.mean()), 2),
                            'max_transaction': float(col_data.max()),
                            'transactions_count': len(col_data)
                        }

            except Exception as e:
                logger.warning(f"Ошибка расчета метрик для колонки {col}: {e}")

        # Анализ категориальных данных
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            try:
                if df[col].nunique() < 100:  # Ограничение для производительности
                    metrics['unique_entities'][col] = int(df[col].nunique())

                    # Специальные метрики для клиентских данных
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['customer', 'user', 'client']):
                        top_customers = df[col].value_counts().head(5)
                        metrics['customer_metrics'][col] = {
                            'unique_customers': int(df[col].nunique()),
                            'total_interactions': len(df),
                            'avg_interactions_per_customer': round(len(df) / df[col].nunique(), 2),
                            'top_customers': convert_to_serializable(top_customers.to_dict())
                        }

            except Exception as e:
                logger.warning(f"Ошибка анализа категориальной колонки {col}: {e}")

        return metrics

    def _calculate_advanced_business_metrics(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Расчет продвинутых бизнес-метрик"""
        advanced = {}

        try:
            # Анализ временных паттернов
            date_cols = self._find_date_columns(df)
            if date_cols:
                date_col = date_cols[0]
                try:
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                    df_clean = df_temp.dropna(subset=[date_col])

                    if len(df_clean) > 1:
                        date_range = df_clean[date_col].max() - df_clean[date_col].min()
                        advanced['time_span_days'] = date_range.days
                        advanced['records_per_day'] = round(len(df_clean) / max(date_range.days, 1), 2)

                        # Анализ активности по дням недели
                        if len(df_clean) > 7:
                            weekday_activity = df_clean[date_col].dt.dayofweek.value_counts().sort_index()
                            advanced['weekday_pattern'] = convert_to_serializable(weekday_activity.to_dict())

                except Exception as e:
                    logger.warning(f"Ошибка анализа временных паттернов: {e}")

            # Анализ качества связей (если есть ID колонки)
            id_cols = [col for col in df.columns if 'id' in col.lower()]
            if id_cols:
                id_col = id_cols[0]
                try:
                    unique_ids = df[id_col].nunique()
                    total_records = len(df)
                    advanced['referential_integrity'] = {
                        'unique_ids': int(unique_ids),
                        'total_records': int(total_records),
                        'integrity_ratio': round(unique_ids / total_records, 3),
                        'potential_duplicates': int(total_records - unique_ids)
                    }
                except Exception as e:
                    logger.warning(f"Ошибка анализа целостности: {e}")

        except Exception as e:
            logger.error(f"Ошибка расчета продвинутых метрик: {e}")

        return advanced

    def _analyze_relationships(self) -> Dict[str, Any]:
        """Анализ связей между таблицами с GPT инсайтами"""
        if not self.df_manager.relations:
            return {
                'question': 'Анализ связей между таблицами',
                'data': [],
                'summary': '🔗 Связи между таблицами не обнаружены. Рекомендуется проверить структуру БД.',
                'analyzed_tables': [],
                'gpt_insights': 'Связи не найдены - анализ целостности данных невозможен'
            }

        relationship_data = []
        for relation in self.df_manager.relations:
            try:
                # Анализируем качество связи
                left_df = self.df_manager.tables[relation.from_table]
                right_df = self.df_manager.tables[relation.to_table]

                # Подсчитываем статистику связи
                left_values = left_df[relation.from_column].dropna()
                right_values = right_df[relation.to_column].dropna()

                left_unique = set(left_values.astype(str))
                right_unique = set(right_values.astype(str))
                common_values = left_unique.intersection(right_unique)

                # Проверяем целостность связи
                left_not_in_right = len(left_unique - right_unique)
                right_not_in_left = len(right_unique - left_unique)
                relationship_strength = len(common_values) / max(len(left_unique), 1) * 100

                # Дополнительная статистика
                cardinality_ratio = len(left_unique) / max(len(right_unique), 1)

                relationship_data.append({
                    'from_table': relation.from_table,
                    'to_table': relation.to_table,
                    'from_column': relation.from_column,
                    'to_column': relation.to_column,
                    'common_values_count': int(len(common_values)),
                    'left_unique_count': int(len(left_unique)),
                    'right_unique_count': int(len(right_unique)),
                    'relationship_strength': round(float(relationship_strength), 2),
                    'integrity_issues': int(left_not_in_right),
                    'reverse_integrity_issues': int(right_not_in_left),
                    'cardinality_ratio': round(float(cardinality_ratio), 3),
                    'relation_type': relation.relation_type,
                    'quality_score': self._calculate_relationship_quality(
                        relationship_strength, left_not_in_right, right_not_in_left
                    )
                })

            except Exception as e:
                logger.error(f"Ошибка анализа связи {relation.from_table}->{relation.to_table}: {e}")
                relationship_data.append({
                    'from_table': relation.from_table,
                    'to_table': relation.to_table,
                    'from_column': relation.from_column,
                    'to_column': relation.to_column,
                    'error': str(e),
                    'quality_score': 0
                })

        if not relationship_data:
            return {
                'question': 'Анализ связей между таблицами',
                'data': [],
                'summary': 'Не удалось проанализировать связи',
                'analyzed_tables': []
            }

        relationships_df = pd.DataFrame(relationship_data)

        # GPT анализ связей
        try:
            gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                df=relationships_df,
                table_name='database_relationships',
                analysis_type='data_quality',
                context={
                    'relationships_count': len(relationship_data),
                    'integrity_issues': sum(r.get('integrity_issues', 0) for r in relationship_data),
                    'avg_strength': relationships_df[
                        'relationship_strength'].mean() if not relationships_df.empty else 0
                }
            )
            gpt_insights = gpt_analysis.get('gpt_analysis', '')
        except Exception as e:
            logger.error(f"Ошибка GPT анализа связей: {e}")
            gpt_insights = 'GPT анализ связей недоступен'

        # Статистика
        if not relationships_df.empty:
            strongest = relationships_df.loc[relationships_df['relationship_strength'].idxmax()]
            weakest = relationships_df.loc[relationships_df['relationship_strength'].idxmin()]
            avg_strength = relationships_df['relationship_strength'].mean()
            total_issues = relationships_df['integrity_issues'].sum()

            summary = f"🔗 **Анализ {len(self.df_manager.relations)} связей между таблицами**\n\n"
            summary += f"📊 **Статистика:**\n"
            summary += f"• Средняя сила связей: {avg_strength:.1f}%\n"
            summary += f"• Самая сильная: {strongest['from_table']} → {strongest['to_table']} ({strongest['relationship_strength']:.1f}%)\n"
            summary += f"• Самая слабая: {weakest['from_table']} → {weakest['to_table']} ({weakest['relationship_strength']:.1f}%)\n"
            summary += f"• Проблем целостности: {total_issues}\n\n"
            summary += f"🤖 **GPT Анализ:**\n{gpt_insights}"
        else:
            summary = "Анализ связей не дал результатов"

        analyzed_tables = list(set([r['from_table'] for r in relationship_data] +
                                   [r['to_table'] for r in relationship_data]))

        return {
            'question': 'Анализ связей между таблицами',
            'data': clean_dataframe_for_json(relationships_df),
            'summary': summary,
            'analyzed_tables': analyzed_tables,
            'gpt_insights': gpt_insights,
            'relationship_stats': {
                'total_relations': len(relationship_data),
                'avg_strength': round(avg_strength, 1) if not relationships_df.empty else 0,
                'total_integrity_issues': int(total_issues) if not relationships_df.empty else 0
            },
            'chart_data': self._prepare_chart_data_safe(relationships_df, 'bar', 'from_table', 'relationship_strength')
        }

    def _calculate_relationship_quality(self, strength: float, left_issues: int, right_issues: int) -> float:
        """Вычисляет качество связи (0-100)"""
        base_score = strength  # Начинаем с силы связи

        # Снижаем за проблемы целостности
        integrity_penalty = (left_issues + right_issues) * 2
        quality_score = max(0, int(base_score - integrity_penalty))

        return round(quality_score, 1)

    def _analyze_aggregations(self, question: str) -> Dict[str, Any]:
        """Анализ с агрегациями и GPT интерпретацией"""
        tables_mentioned = self._extract_mentioned_tables(question)
        if not tables_mentioned:
            if self.df_manager.tables:
                tables_mentioned = [max(self.df_manager.tables.items(), key=lambda x: len(x[1]))[0]]
            else:
                return {
                    'question': question,
                    'data': [],
                    'summary': 'Нет доступных таблиц для анализа',
                    'analyzed_tables': []
                }

        results = []
        aggregated_data = []

        for table_name in tables_mentioned:
            df = self.df_manager.tables[table_name]

            # Числовые агрегации
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                agg_funcs = ['count', 'sum', 'mean', 'min', 'max', 'std']
                try:
                    agg_result = df[numeric_cols].agg(agg_funcs).round(2)

                    # Преобразуем в удобный формат
                    for col in numeric_cols:
                        col_data = {
                            'table': table_name,
                            'column': col,
                            'type': 'numeric'
                        }

                        for func in agg_funcs:
                            try:
                                value = agg_result.loc[func, col]
                                if pd.isna(value) or np.isinf(value):
                                    value = 0
                                col_data[func] = float(value)
                            except:
                                col_data[func] = 0

                        # Добавляем дополнительные метрики
                        try:
                            col_series = df[col].dropna()
                            if len(col_series) > 0:
                                col_data['median'] = float(col_series.median())
                                col_data['q25'] = float(col_series.quantile(0.25))
                                col_data['q75'] = float(col_series.quantile(0.75))
                                col_data['non_zero_count'] = int((col_series != 0).sum())
                        except:
                            col_data.update({'median': 0, 'q25': 0, 'q75': 0, 'non_zero_count': 0})

                        aggregated_data.append(col_data)

                    results.append({
                        'table': table_name,
                        'type': 'numeric_aggregation',
                        'columns_processed': len(numeric_cols),
                        'data': convert_to_serializable(agg_result)
                    })
                except Exception as e:
                    logger.error(f"Ошибка числовой агрегации в {table_name}: {e}")

            # Категориальные агрегации
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols[:5]:  # Ограичиваем количество
                    try:
                        if df[col].nunique() < 50:  # Избегаем колонок с слишком большим разнообразием
                            value_counts = df[col].value_counts().head(10)
                            if len(value_counts) > 0:
                                for value, count in value_counts.items():
                                    aggregated_data.append({
                                        'table': table_name,
                                        'column': col,
                                        'type': 'categorical',
                                        'value': str(value),
                                        'count': int(count),
                                        'percentage': round(float(count / len(df) * 100), 2)
                                    })

                                results.append({
                                    'table': table_name,
                                    'type': 'categorical_aggregation',
                                    'column': col,
                                    'unique_values': int(df[col].nunique()),
                                    'top_values': convert_to_serializable(value_counts.head(5).to_dict())
                                })
                    except Exception as e:
                        logger.error(f"Ошибка категориальной агрегации {col} в {table_name}: {e}")

        # Создаем итоговый DataFrame
        if aggregated_data:
            main_result = pd.DataFrame(aggregated_data)

            # GPT анализ агрегаций
            try:
                gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                    df=main_result,
                    table_name='aggregation_results',
                    analysis_type='statistical_insights',
                    context={
                        'question': question,
                        'tables_analyzed': tables_mentioned,
                        'aggregation_results': results[:3]  # Первые 3 результата для контекста
                    }
                )
                gpt_insights = gpt_analysis.get('gpt_analysis', '')
            except Exception as e:
                logger.error(f"Ошибка GPT анализа агрегаций: {e}")
                gpt_insights = 'GPT анализ агрегаций недоступен'

            summary = f"📊 **Агрегированный анализ {len(tables_mentioned)} таблиц**\n\n"
            summary += f"📈 Проанализировано {len([r for r in results if r['type'] == 'numeric_aggregation'])} числовых метрик\n"
            summary += f"📋 Проанализировано {len([r for r in results if r['type'] == 'categorical_aggregation'])} категориальных переменных\n\n"
            summary += f"🤖 **GPT Инсайты:**\n{gpt_insights}"
        else:
            main_result = pd.DataFrame()
            summary = "Не удалось выполнить агрегацию данных"
            gpt_insights = ''

        return {
            'question': question,
            'data': clean_dataframe_for_json(main_result),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'gpt_insights': gpt_insights,
            'detailed_results': convert_to_serializable(results),
            'aggregation_stats': {
                'tables_processed': len(tables_mentioned),
                'total_aggregations': len(aggregated_data),
                'numeric_aggregations': len([d for d in aggregated_data if d['type'] == 'numeric']),
                'categorical_aggregations': len([d for d in aggregated_data if d['type'] == 'categorical'])
            },
            'chart_data': self._create_aggregation_chart(main_result)
        }

    def _create_aggregation_chart(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Создает данные для графика агрегации"""
        try:
            if df.empty:
                return None

            # Ищем подходящие колонки для графика
            if 'type' in df.columns and 'count' in df.columns:
                # Группируем по типу
                grouped = df.groupby('type')['count'].sum().reset_index()
                return self._prepare_chart_data_safe(grouped, 'bar', 'type', 'count')

            elif 'table' in df.columns and 'count' in df.columns:
                # Группируем по таблицам
                grouped = df.groupby('table')['count'].sum().reset_index()
                return self._prepare_chart_data_safe(grouped, 'bar', 'table', 'count')

            elif len(df.columns) >= 2:
                # Используем первые две колонки
                return self._prepare_chart_data_safe(df, 'bar', df.columns[0], df.columns[1])

            return None

        except Exception as e:
            logger.error(f"Ошибка создания aggregation chart: {e}")
            return None

    def _analyze_trends(self, question: str) -> Dict[str, Any]:
        """Временной анализ с GPT прогнозами"""
        trend_results = []

        for table_name, df in self.df_manager.tables.items():
            # Ищем колонки с датами
            date_cols = self._find_date_columns(df)
            if date_cols:
                for date_col in date_cols[:2]:  # Максимум 2 даты на таблицу
                    try:
                        df_copy = df.copy()
                        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')

                        # Убираем строки с некорректными датами
                        df_clean = df_copy.dropna(subset=[date_col])

                        if len(df_clean) > 5:  # Минимум данных для тренда
                            # Группируем по периодам
                            period_type = self._determine_period_type(df_clean[date_col])
                            df_clean['period'] = df_clean[date_col].dt.to_period(period_type)

                            # Базовая статистика по периодам
                            trend_data = df_clean.groupby('period').agg({
                                date_col: 'count'
                            }).rename(columns={date_col: 'count'}).reset_index()
                            trend_data['period_str'] = trend_data['period'].astype(str)

                            # Добавляем агрегации по числовым колонкам
                            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                numeric_trends = df_clean.groupby('period')[numeric_cols].agg(['sum', 'mean']).round(2)
                                # Добавляем к trend_data
                                for col in numeric_cols[:3]:  # Первые 3 числовые колонки
                                    try:
                                        trend_data[f'{col}_sum'] = numeric_trends[(col, 'sum')].values
                                        trend_data[f'{col}_avg'] = numeric_trends[(col, 'mean')].values
                                    except:
                                        continue

                            # Анализ тренда
                            if len(trend_data) > 1:
                                first_val = int(trend_data['count'].iloc[0])
                                last_val = int(trend_data['count'].iloc[-1])
                                trend_direction = "рост" if last_val > first_val else "падение"
                                trend_percent = abs((last_val - first_val) / first_val * 100) if first_val > 0 else 0

                                # Дополнительная статистика
                                volatility = trend_data['count'].std() / trend_data['count'].mean() * 100 if trend_data[
                                                                                                                 'count'].mean() > 0 else 0
                                peak_period = trend_data.loc[trend_data['count'].idxmax(), 'period_str']
                                min_period = trend_data.loc[trend_data['count'].idxmin(), 'period_str']
                            else:
                                trend_direction = "недостаточно данных"
                                trend_percent = 0
                                volatility = 0
                                peak_period = trend_data['period_str'].iloc[0] if len(trend_data) > 0 else 'N/A'
                                min_period = peak_period

                            # Временной диапазон
                            min_date = df_clean[date_col].min()
                            max_date = df_clean[date_col].max()
                            try:
                                date_range = f"{min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}"
                                duration_days = (max_date - min_date).days
                            except:
                                date_range = f"{str(min_date)[:10]} - {str(max_date)[:10]}"
                                duration_days = 0

                            trend_results.append({
                                'table': table_name,
                                'date_column': date_col,
                                'trend_data': clean_dataframe_for_json(trend_data),
                                'total_records': int(len(df_clean)),
                                'periods_count': int(len(trend_data)),
                                'trend_direction': trend_direction,
                                'trend_percent': round(float(trend_percent), 1),
                                'volatility': round(float(volatility), 1),
                                'peak_period': peak_period,
                                'low_period': min_period,
                                'date_range': date_range,
                                'duration_days': int(duration_days),
                                'period_type': period_type,
                                'avg_records_per_period': round(len(df_clean) / len(trend_data), 1)
                            })

                    except Exception as e:
                        logger.error(f"Ошибка анализа трендов для {table_name}.{date_col}: {e}")

        if trend_results:
            # Берем самый информативный тренд
            main_trend = max(trend_results, key=lambda x: x['total_records'])

            # GPT анализ трендов
            try:
                gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                    df=pd.DataFrame(trend_results),
                    table_name='trend_analysis',
                    analysis_type='predictive_analysis',
                    context={
                        'question': question,
                        'main_trend': main_trend,
                        'total_trends': len(trend_results)
                    }
                )
                gpt_insights = gpt_analysis.get('gpt_analysis', '')
            except Exception as e:
                logger.error(f"Ошибка GPT анализа трендов: {e}")
                gpt_insights = 'GPT анализ трендов недоступен'

            summary = f"📈 **Временной анализ найден в {len(trend_results)} таблицах**\n\n"
            summary += f"🎯 **Основной тренд** (таблица '{main_trend['table']}'):\n"
            summary += f"• Период: {main_trend['date_range']}\n"
            summary += f"• Направление: {main_trend['trend_direction']} на {main_trend['trend_percent']:.1f}%\n"
            summary += f"• Волатильность: {main_trend['volatility']:.1f}%\n"
            summary += f"• Пик активности: {main_trend['peak_period']}\n"
            summary += f"• Минимум активности: {main_trend['low_period']}\n\n"
            summary += f"🤖 **GPT Прогнозы и Инсайты:**\n{gpt_insights}"

            main_data = main_trend['trend_data']
        else:
            main_data = []
            summary = "📅 Временные данные для анализа трендов не найдены"
            gpt_insights = ''

        analyzed_tables = [r['table'] for r in trend_results]

        return {
            'question': question,
            'data': main_data,
            'summary': summary,
            'analyzed_tables': analyzed_tables,
            'gpt_insights': gpt_insights,
            'all_trends': trend_results,
            'trend_stats': {
                'tables_with_trends': len(trend_results),
                'total_periods_analyzed': sum(t['periods_count'] for t in trend_results),
                'avg_trend_strength': round(np.mean([abs(t['trend_percent']) for t in trend_results]), 1) if trend_results else 0
            },
            'chart_data': self._prepare_chart_data_safe_from_list(main_data, 'line', 'period_str',
                                                                  'count') if main_data else None
        }

    def _prepare_chart_data_safe_from_list(self, data_list: List[Dict], chart_type: str,
                                           x_key: str, y_key: str) -> Optional[Dict[str, Any]]:
        """Безопасная подготовка данных для графиков из списка словарей"""
        try:
            if not data_list or not isinstance(data_list, list):
                return None

            # Проверяем наличие ключей в первом элементе
            if not data_list or x_key not in data_list[0] or y_key not in data_list[0]:
                return None

            chart_data = {
                'chart_type': chart_type,
                'x_column': x_key,
                'y_column': y_key,
                'data': {
                    'labels': [str(item.get(x_key, '')) for item in data_list[:50]],
                    'values': [float(item.get(y_key, 0)) for item in data_list[:50]]
                }
            }

            return chart_data

        except Exception as e:
            logger.error(f"Ошибка подготовки chart_data из списка: {e}")
            return None

    def _determine_period_type(self, date_series: pd.Series) -> str:
        """Определяет подходящий тип периода для группировки"""
        date_range = date_series.max() - date_series.min()

        if date_range.days <= 31:
            return 'D'  # Дни
        elif date_range.days <= 365:
            return 'W'  # Недели
        elif date_range.days <= 365 * 3:
            return 'M'  # Месяцы
        else:
            return 'Y'  # Годы

    def _analyze_correlations(self, question: str) -> Dict[str, Any]:
        """Анализ корреляций с GPT интерпретацией"""
        tables_mentioned = self._extract_mentioned_tables(question)
        if not tables_mentioned:
            tables_mentioned = list(self.df_manager.tables.keys())

        all_correlations = []
        correlation_matrices = {}

        for table_name in tables_mentioned:
            df = self.df_manager.tables[table_name]
            correlations = self._analyze_correlations_single_table(df, table_name)
            all_correlations.extend(correlations)

            # Сохраняем матрицу корреляций для GPT анализа
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                try:
                    corr_matrix = df[numeric_cols].corr()
                    correlation_matrices[table_name] = convert_to_serializable(corr_matrix.to_dict())
                except:
                    pass

        if all_correlations:
            # Создаем DataFrame с корреляциями
            corr_df = pd.DataFrame(all_correlations)

            # Сортируем по силе корреляции
            corr_df['abs_correlation'] = abs(corr_df['correlation'])
            corr_df = corr_df.sort_values('abs_correlation', ascending=False)
            corr_df = corr_df.drop('abs_correlation', axis=1)

            # Классификация корреляций
            strong_correlations = corr_df[abs(corr_df['correlation']) > 0.7]
            moderate_correlations = corr_df[(abs(corr_df['correlation']) > 0.4) & (abs(corr_df['correlation']) <= 0.7)]
            weak_correlations = corr_df[abs(corr_df['correlation']) <= 0.4]

            # GPT анализ корреляций
            try:
                gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                    df=corr_df,
                    table_name='correlation_analysis',
                    analysis_type='statistical_insights',
                    context={
                        'question': question,
                        'strong_count': len(strong_correlations),
                        'moderate_count': len(moderate_correlations),
                        'correlation_matrices': correlation_matrices,
                        'top_correlations': all_correlations[:5]
                    }
                )
                gpt_insights = gpt_analysis.get('gpt_analysis', '')
            except Exception as e:
                logger.error(f"Ошибка GPT анализа корреляций: {e}")
                gpt_insights = 'GPT анализ корреляций недоступен'

            summary = f"🔗 **Анализ корреляций в {len(tables_mentioned)} таблицах**\n\n"
            summary += f"📊 **Найдено корреляций:**\n"
            summary += f"• Сильных (>0.7): {len(strong_correlations)}\n"
            summary += f"• Умеренных (0.4-0.7): {len(moderate_correlations)}\n"
            summary += f"• Слабых (<0.4): {len(weak_correlations)}\n\n"

            if len(strong_correlations) > 0:
                top_correlation = strong_correlations.iloc[0]
                summary += f"🎯 **Самая сильная корреляция:** {top_correlation['column1']} ↔ {top_correlation['column2']} "
                summary += f"({top_correlation['correlation']:.3f}) в таблице {top_correlation['table']}\n\n"

            summary += f"🤖 **GPT Инсайты:**\n{gpt_insights}"
        else:
            corr_df = pd.DataFrame()
            summary = "🔗 Значимые корреляции не найдены"
            gpt_insights = ''

        return {
            'question': question,
            'data': clean_dataframe_for_json(corr_df),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'gpt_insights': gpt_insights,
            'correlation_stats': {
                'total_correlations': len(all_correlations),
                'strong_correlations': len(strong_correlations) if all_correlations else 0,
                'moderate_correlations': len(moderate_correlations) if all_correlations else 0,
                'tables_analyzed': len(tables_mentioned)
            },
            'correlation_matrices': correlation_matrices,
            'chart_data': self._prepare_chart_data_safe(corr_df, 'scatter', 'column1',
                                                        'correlation') if not corr_df.empty else None
        }

    def _compare_columns_in_table(self, table_name: str, question: str) -> Dict[str, Any]:
        """Сравнение колонок внутри одной таблицы"""
        df = self.df_manager.tables[table_name]

        # Получаем числовые колонки для сравнения
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) < 2:
            return {
                'question': question,
                'data': [],
                'summary': f'В таблице {table_name} недостаточно числовых колонок для сравнения',
                'analyzed_tables': [table_name]
            }

        comparison_data = []

        # Сравниваем первые несколько числовых колонок
        for col in numeric_cols[:5]:
            try:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    comparison_data.append({
                        'column': col,
                        'mean': round(float(col_data.mean()), 2),
                        'median': round(float(col_data.median()), 2),
                        'std': round(float(col_data.std()), 2),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'range': float(col_data.max() - col_data.min()),
                        'count': int(len(col_data))
                    })
            except Exception as e:
                logger.error(f"Ошибка сравнения колонки {col}: {e}")

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)

            # Находим колонку с максимальным средним значением
            max_mean_col = comparison_df.loc[comparison_df['mean'].idxmax(), 'column']
            min_mean_col = comparison_df.loc[comparison_df['mean'].idxmin(), 'column']

            summary = f"📊 Сравнение числовых колонок в таблице '{table_name}'. "
            summary += f"Максимальное среднее значение: {max_mean_col}. "
            summary += f"Минимальное среднее значение: {min_mean_col}."
        else:
            comparison_df = pd.DataFrame()
            summary = "Не удалось сравнить колонки"

        return {
            'question': question,
            'data': clean_dataframe_for_json(comparison_df),
            'summary': summary,
            'analyzed_tables': [table_name],
            'chart_data': self._prepare_chart_data_safe(comparison_df, 'bar', 'column', 'mean')
        }


    def _analyze_comparison(self, question: str) -> Dict[str, Any]:
        """Анализ сравнения между таблицами с GPT выводами"""
        tables_mentioned = self._extract_mentioned_tables(question)

        if len(tables_mentioned) < 2:
            if len(tables_mentioned) == 1:
                return self._compare_columns_in_table(tables_mentioned[0], question)
            else:
                if len(self.df_manager.tables) >= 2:
                    largest_tables = sorted(self.df_manager.tables.items(),
                                            key=lambda x: len(x[1]), reverse=True)[:2]
                    tables_mentioned = [t[0] for t in largest_tables]
                else:
                    return {
                        'question': question,
                        'data': [],
                        'summary': 'Недостаточно таблиц для сравнения',
                        'analyzed_tables': []
                    }

        comparison_data = []
        detailed_comparison = {}

        for table_name in tables_mentioned:
            df = self.df_manager.tables[table_name]

            # Расширенная статистика таблицы
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            stats = {
                'table': table_name,
                'rows': int(len(df)),
                'columns': int(len(df.columns)),
                'numeric_columns': int(len(numeric_cols)),
                'categorical_columns': int(len(categorical_cols)),
                'null_values': int(df.isnull().sum().sum()),
                'null_percentage': round(float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100), 2),
                'duplicate_rows': int(df.duplicated().sum()),
                'memory_mb': round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 2)
            }

            # Добавляем статистику по числовым колонкам
            if len(numeric_cols) > 0:
                try:
                    numeric_desc = df[numeric_cols].describe()
                    stats.update({
                        'avg_mean': round(float(numeric_desc.loc['mean'].mean()), 2),
                        'avg_std': round(float(numeric_desc.loc['std'].mean()), 2),
                        'avg_min': round(float(numeric_desc.loc['min'].mean()), 2),
                        'avg_max': round(float(numeric_desc.loc['max'].mean()), 2)
                    })
                except:
                    stats.update({'avg_mean': 0, 'avg_std': 0, 'avg_min': 0, 'avg_max': 0})

            # Дополнительные метрики качества данных
            stats.update({
                'data_density': round(float(len(df.drop_duplicates()) / len(df) * 100), 2) if len(df) > 0 else 0,
                'avg_column_completeness': round(float((1 - df.isnull().mean().mean()) * 100), 2),
                'schema_complexity': len(df.columns) * len(numeric_cols) / max(len(df.columns), 1)
            })

            comparison_data.append(stats)
            detailed_comparison[table_name] = stats

        comparison_df = pd.DataFrame(comparison_data)

        # Анализ различий
        differences_analysis = {}
        if len(comparison_data) >= 2:
            for i in range(len(comparison_data)):
                for j in range(i + 1, len(comparison_data)):
                    table1, table2 = comparison_data[i], comparison_data[j]
                    pair_key = f"{table1['table']}_vs_{table2['table']}"

                    differences_analysis[pair_key] = {
                        'size_ratio': table1['rows'] / max(table2['rows'], 1),
                        'complexity_diff': abs(table1['columns'] - table2['columns']),
                        'quality_diff': abs(table1['avg_column_completeness'] - table2['avg_column_completeness']),
                        'memory_ratio': table1['memory_mb'] / max(table2['memory_mb'], 1)
                    }

        # GPT анализ сравнения
        try:
            gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                df=comparison_df,
                table_name='table_comparison',
                analysis_type='business_insights',
                context={
                    'question': question,
                    'tables_compared': tables_mentioned,
                    'differences_analysis': differences_analysis,
                    'detailed_stats': detailed_comparison
                }
            )
            gpt_insights = gpt_analysis.get('gpt_analysis', '')
        except Exception as e:
            logger.error(f"Ошибка GPT анализа сравнения: {e}")
            gpt_insights = 'GPT анализ сравнения недоступен'

        # Создание сводки
        if len(comparison_data) >= 2:
            summary = f"⚖️ **Сравнение таблиц: {' vs '.join(tables_mentioned)}**\n\n"

            # Основные различия
            table1, table2 = comparison_data[0], comparison_data[1]
            summary += f"📊 **Ключевые различия:**\n"
            summary += f"• Размер: {table1['table']} ({table1['rows']:,} строк) vs {table2['table']} ({table2['rows']:,} строк)\n"
            summary += f"• Сложность: {table1['columns']} vs {table2['columns']} колонок\n"
            summary += f"• Качество данных: {table1['avg_column_completeness']:.1f}% vs {table2['avg_column_completeness']:.1f}%\n"
            summary += f"• Использование памяти: {table1['memory_mb']:.1f} vs {table2['memory_mb']:.1f} MB\n\n"
            summary += f"🤖 **GPT Анализ:**\n{gpt_insights}"
        else:
            summary = "Недостаточно данных для сравнения"

        return {
            'question': question,
            'data': clean_dataframe_for_json(comparison_df),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'gpt_insights': gpt_insights,
            'detailed_comparison': detailed_comparison,
            'differences_analysis': differences_analysis,
            'chart_data': self._prepare_chart_data_safe(comparison_df, 'bar', 'table', 'rows')
        }

    def _analyze_anomalies(self, question: str) -> Dict[str, Any]:
        """Специальный анализ аномалий с GPT диагностикой"""
        tables_mentioned = self._extract_mentioned_tables(question)
        if not tables_mentioned:
            tables_mentioned = list(self.df_manager.tables.keys())

        all_anomalies = []
        anomaly_details = {}

        for table_name in tables_mentioned:
            df = self.df_manager.tables[table_name]
            anomalies = self._detect_anomalies_dataframe(df)

            for anomaly in anomalies:
                anomaly['table'] = table_name
                all_anomalies.append(anomaly)

            # Дополнительные проверки аномалий
            additional_anomalies = self._detect_advanced_anomalies(df, table_name)
            all_anomalies.extend(additional_anomalies)

            # Сохраняем детали для GPT
            anomaly_details[table_name] = {
                'total_anomalies': len(anomalies) + len(additional_anomalies),
                'anomaly_types': list(set([a['type'] for a in anomalies + additional_anomalies])),
                'severity': self._calculate_anomaly_severity(anomalies + additional_anomalies, df)
            }

        if all_anomalies:
            anomalies_df = pd.DataFrame(all_anomalies)
            total_anomalies = sum(a.get('count', 1) for a in all_anomalies)

            # Группировка по типам аномалий
            anomaly_types = {}
            for anomaly in all_anomalies:
                anom_type = anomaly.get('type', 'outlier')
                if anom_type not in anomaly_types:
                    anomaly_types[anom_type] = []
                anomaly_types[anom_type].append(anomaly)

            # GPT анализ аномалий
            try:
                gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                    df=anomalies_df,
                    table_name='anomaly_analysis',
                    analysis_type='data_quality',
                    context={
                        'question': question,
                        'anomaly_details': anomaly_details,
                        'total_anomalies': total_anomalies,
                        'anomaly_types': anomaly_types,
                        'tables_with_issues': len([t for t in anomaly_details.values() if t['total_anomalies'] > 0])
                    }
                )
                gpt_insights = gpt_analysis.get('gpt_analysis', '')
            except Exception as e:
                logger.error(f"Ошибка GPT анализа аномалий: {e}")
                gpt_insights = 'GPT анализ аномалий недоступен'

            # Приоритизация аномалий
            high_priority = [a for a in all_anomalies if a.get('severity', 'medium') == 'high']
            medium_priority = [a for a in all_anomalies if a.get('severity', 'medium') == 'medium']

            summary = f"🚨 **Обнаружено {len(all_anomalies)} типов аномалий в {len(tables_mentioned)} таблицах**\n\n"
            summary += f"📊 **Статистика аномалий:**\n"
            summary += f"• Высокий приоритет: {len(high_priority)}\n"
            summary += f"• Средний приоритет: {len(medium_priority)}\n"
            summary += f"• Общее количество аномальных записей: {total_anomalies:,}\n\n"

            if anomaly_types:
                summary += f"🔍 **Типы найденных аномалий:**\n"
                for anom_type, items in anomaly_types.items():
                    summary += f"• {anom_type}: {len(items)} случаев\n"
                summary += "\n"

            summary += f"🤖 **GPT Диагностика:**\n{gpt_insights}"
        else:
            anomalies_df = pd.DataFrame()
            summary = "✅ Аномалии в данных не обнаружены - данные выглядят корректными"
            gpt_insights = ''

        return {
            'question': question,
            'data': clean_dataframe_for_json(anomalies_df),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'gpt_insights': gpt_insights,
            'anomaly_details': anomaly_details,
            'anomaly_stats': {
                'total_anomaly_types': len(all_anomalies),
                'total_anomaly_records': sum(a.get('count', 1) for a in all_anomalies),
                'tables_with_anomalies': len([t for t in anomaly_details.values() if t['total_anomalies'] > 0]),
                'high_priority_count': len([a for a in all_anomalies if a.get('severity', 'medium') == 'high'])
            },
            'chart_data': self._prepare_chart_data_safe(anomalies_df, 'bar', 'table',
                                                        'count') if not anomalies_df.empty else None
        }

    def _detect_advanced_anomalies(self, df: pd.DataFrame, table_name: str) -> List[Dict[str, Any]]:
        """Расширенное обнаружение аномалий"""
        anomalies = []

        try:
            # Проверка на подозрительно одинаковые значения
            for col in df.columns:
                if df[col].dtype in ['object']:
                    value_counts = df[col].value_counts()
                    if len(value_counts) > 0:
                        top_value_freq = value_counts.iloc[0]
                        if top_value_freq / len(df) > 0.9:  # 90% одинаковых значений
                            anomalies.append({
                                'type': 'excessive_repetition',
                                'column': col,
                                'count': int(top_value_freq),
                                'percentage': round(float(top_value_freq / len(df) * 100), 2),
                                'description': f'Значение "{value_counts.index[0]}" встречается в {top_value_freq} из {len(df)} записей',
                                'severity': 'medium'
                            })

            # Проверка на подозрительные паттерны в числовых данных
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].count() > 10:
                    # Проверка на слишком много нулей
                    zero_count = (df[col] == 0).sum()
                    if zero_count / len(df) > 0.5:  # Больше 50% нулей
                        anomalies.append({
                            'type': 'excessive_zeros',
                            'column': col,
                            'count': int(zero_count),
                            'percentage': round(float(zero_count / len(df) * 100), 2),
                            'description': f'Колонка содержит {zero_count} нулевых значений из {len(df)}',
                            'severity': 'low'
                        })

                    # Проверка на подозрительно круглые числа
                    if df[col].dtype in ['int64', 'float64']:
                        round_numbers = df[col][df[col] % 10 == 0]
                        if len(round_numbers) / len(df[col].dropna()) > 0.8:  # 80% круглых чисел
                            anomalies.append({
                                'type': 'excessive_round_numbers',
                                'column': col,
                                'count': len(round_numbers),
                                'percentage': round(float(len(round_numbers) / len(df[col].dropna()) * 100), 2),
                                'description': f'Подозрительно много круглых чисел в колонке',
                                'severity': 'low'
                            })

            # Проверка на странные временные паттерны
            date_cols = self._find_date_columns(df)
            for date_col in date_cols:
                try:
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                    df_dates = df_temp.dropna(subset=[date_col])

                    if len(df_dates) > 10:
                        # Проверка на все записи в один день
                        dates_only = df_dates[date_col].dt.date
                        if dates_only.nunique() == 1:
                            anomalies.append({
                                'type': 'single_date_anomaly',
                                'column': date_col,
                                'count': len(df_dates),
                                'description': f'Все {len(df_dates)} записей имеют одинаковую дату: {dates_only.iloc[0]}',
                                'severity': 'medium'
                            })
                except:
                    continue

        except Exception as e:
            logger.error(f"Ошибка расширенного поиска аномалий в {table_name}: {e}")

        return anomalies

    def _calculate_anomaly_severity(self, anomalies: List[Dict], df: pd.DataFrame) -> str:
        """Вычисляет общую серьезность аномалий"""
        if not anomalies:
            return 'none'

        high_count = len([a for a in anomalies if a.get('severity') == 'high'])
        medium_count = len([a for a in anomalies if a.get('severity') == 'medium'])

        if high_count > 0:
            return 'high'
        elif medium_count > 2:
            return 'high'
        elif medium_count > 0:
            return 'medium'
        else:
            return 'low'

    def _prepare_chart_data_safe(self, df: pd.DataFrame, chart_type: str,
                                 x_col: str, y_col: Optional[str]) -> Optional[Dict]:
        """Безопасная подготовка данных для графиков"""
        try:
            if df.empty or x_col not in df.columns:
                return None

            # Ограничиваем количество точек для производительности
            df_limited = df.head(50)

            chart_data = {
                'chart_type': chart_type,
                'x_column': x_col,
                'y_column': y_col
            }

            if chart_type == 'histogram':
                if pd.api.types.is_numeric_dtype(df_limited[x_col]):
                    chart_data['data'] = {'values': df_limited[x_col].dropna().astype(float).tolist()}
                else:
                    value_counts = df_limited[x_col].value_counts().head(10)
                    chart_data['data'] = {
                        'labels': value_counts.index.astype(str).tolist(),
                        'values': value_counts.values.tolist()
                    }

            elif chart_type == 'bar' and y_col and y_col in df_limited.columns:
                chart_data['data'] = {
                    'labels': df_limited[x_col].astype(str).tolist(),
                    'values': df_limited[y_col].astype(float).tolist()
                }

            elif chart_type == 'scatter' and y_col and y_col in df_limited.columns:
                chart_data['data'] = {
                    'points': [
                        {'x': float(x), 'y': float(y)}
                        for x, y in zip(df_limited[x_col].astype(float), df_limited[y_col].astype(float))
                        if pd.notna(x) and pd.notna(y)
                    ]
                }

            elif chart_type == 'line' and y_col and y_col in df_limited.columns:
                chart_data['data'] = {
                    'labels': df_limited[x_col].astype(str).tolist(),
                    'values': df_limited[y_col].astype(float).tolist()
                }

            return chart_data

        except Exception as e:
            logger.error(f"Ошибка подготовки chart_data: {e}")
            return None

    def _analyze_data_quality(self, question: str) -> Dict[str, Any]:
        """Комплексный анализ качества данных"""
        tables_mentioned = self._extract_mentioned_tables(question)
        if not tables_mentioned:
            tables_mentioned = list(self.df_manager.tables.keys())

        quality_results = []

        for table_name in tables_mentioned:
            df = self.df_manager.tables[table_name]

            # Базовые метрики качества
            total_cells = len(df) * len(df.columns)
            null_cells = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()

            # Расширенный анализ качества
            completeness = (1 - null_cells / total_cells) * 100
            uniqueness = (1 - duplicate_rows / len(df)) * 100
            consistency = self._calculate_consistency_score(df)
            validity = self._calculate_validity_score(df)

            # Общий балл качества
            quality_score = (completeness * 0.3 + uniqueness * 0.2 +
                             consistency * 0.25 + validity * 0.25)

            quality_results.append({
                'table': table_name,
                'quality_score': round(quality_score, 1),
                'completeness': round(completeness, 1),
                'uniqueness': round(uniqueness, 1),
                'consistency': round(consistency, 1),
                'validity': round(validity, 1),
                'total_records': len(df),
                'null_values': int(null_cells),
                'duplicate_records': int(duplicate_rows)
            })

        quality_df = pd.DataFrame(quality_results)

        # GPT анализ качества данных
        try:
            avg_quality = quality_df['quality_score'].mean() if not quality_df.empty else 0
            gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                df=quality_df,
                table_name='data_quality_analysis',
                analysis_type='data_quality',
                context={
                    'question': question,
                    'avg_quality': avg_quality,
                    'quality_results': quality_results
                }
            )
            gpt_insights = gpt_analysis.get('gpt_analysis', '')
        except Exception as e:
            logger.error(f"Ошибка GPT анализа качества: {e}")
            gpt_insights = 'GPT анализ качества данных недоступен'

        if not quality_df.empty:
            avg_quality = quality_df['quality_score'].mean()
            summary = f"🔍 **Анализ качества данных: средний балл {avg_quality:.1f}/100**\n\n"

            best_table = quality_df.loc[quality_df['quality_score'].idxmax()]
            worst_table = quality_df.loc[quality_df['quality_score'].idxmin()]

            summary += f"✅ **Лучшая таблица:** {best_table['table']} ({best_table['quality_score']:.1f}/100)\n"
            summary += f"⚠️ **Требует внимания:** {worst_table['table']} ({worst_table['quality_score']:.1f}/100)\n\n"
            summary += f"🤖 **GPT Рекомендации:**\n{gpt_insights}"
        else:
            summary = "Данные для анализа качества не найдены"

        return {
            'question': question,
            'data': clean_dataframe_for_json(quality_df),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'gpt_insights': gpt_insights,
            'quality_stats': {
                'avg_quality_score': round(quality_df['quality_score'].mean(), 1) if not quality_df.empty else 0,
                'tables_analyzed': len(quality_results),
                'high_quality_tables': len(
                    quality_df[quality_df['quality_score'] >= 80]) if not quality_df.empty else 0,
                'low_quality_tables': len(quality_df[quality_df['quality_score'] < 60]) if not quality_df.empty else 0
            },
            'chart_data': self._prepare_chart_data_safe(quality_df, 'bar', 'table', 'quality_score')
        }

    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Вычисляет балл консистентности данных"""
        consistency_score = 100.0

        try:
            # Проверка форматов в текстовых колонках
            for col in df.select_dtypes(include=['object']).columns:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) > 5:  # Только если достаточно данных
                    # Проверка на смешанные регистры
                    mixed_case = sum(1 for val in unique_vals
                                     if isinstance(val, str) and val != val.lower() and val != val.upper())
                    if mixed_case > len(unique_vals) * 0.1:
                        consistency_score -= 5

        except Exception as e:
            logger.warning(f"Ошибка расчета консистентности: {e}")
            consistency_score = 90  # Дефолтное значение при ошибке

        return max(0, consistency_score)

    def _calculate_validity_score(self, df: pd.DataFrame) -> float:
        """Вычисляет балл валидности данных"""
        validity_score = 100.0

        try:
            # Проверка числовых колонок на отрицательные значения где они не ожидаются
            for col in df.select_dtypes(include=[np.number]).columns:
                if any(keyword in col.lower() for keyword in ['age', 'count', 'quantity', 'price']):
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        validity_score -= (negative_count / len(df)) * 20

            # Проверка на бесконечные значения
            for col in df.select_dtypes(include=[np.number]).columns:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    validity_score -= (inf_count / len(df)) * 30

        except Exception as e:
            logger.warning(f"Ошибка расчета валидности: {e}")
            validity_score = 95  # Дефолтное значение при ошибке

        return max(0, validity_score)

    def _analyze_statistical_insights(self, question: str) -> Dict[str, Any]:
        """Углубленный статистический анализ с GPT интерпретацией"""
        tables_mentioned = self._extract_mentioned_tables(question)
        if not tables_mentioned:
            tables_mentioned = [max(self.df_manager.tables.items(), key=lambda x: len(x[1]))[0]]

        main_table = tables_mentioned[0]
        df = self.df_manager.tables[main_table]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {
                'question': question,
                'data': [],
                'summary': 'В данных отсутствуют числовые колонки для статистического анализа',
                'analyzed_tables': [main_table]
            }

        # Расширенная статистика
        statistical_results = []

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 5:
                try:
                    # Базовая статистика
                    basic_stats = {
                        'column': col,
                        'count': len(col_data),
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'range': float(col_data.max() - col_data.min())
                    }

                    # Расширенная статистика
                    basic_stats.update({
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis()),
                        'coefficient_variation': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0,
                        'outliers_count': len(self._detect_outliers_iqr(col_data)),
                        'distribution_type': self._classify_distribution(col_data.skew(), col_data.kurtosis()),
                        'percentiles': {
                            '25th': float(col_data.quantile(0.25)),
                            '75th': float(col_data.quantile(0.75)),
                            '95th': float(col_data.quantile(0.95))
                        }
                    })

                    statistical_results.append(basic_stats)

                except Exception as e:
                    logger.warning(f"Ошибка статистического анализа для {col}: {e}")

        stats_df = pd.DataFrame(statistical_results)

        # Корреляционный анализ
        correlation_matrix = {}
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                correlation_matrix = convert_to_serializable(corr_matrix.to_dict())
            except:
                pass

        # GPT анализ статистических находок
        try:
            gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                df=stats_df,
                table_name=main_table,
                analysis_type='statistical_insights',
                context={
                    'question': question,
                    'statistical_results': statistical_results[:3],  # Первые 3 для контекста
                    'correlation_matrix': correlation_matrix
                }
            )
            gpt_insights = gpt_analysis.get('gpt_analysis', '')
        except Exception as e:
            logger.error(f"Ошибка GPT статистического анализа: {e}")
            gpt_insights = 'GPT анализ статистических данных недоступен'

        summary = f"📈 **Углубленный статистический анализ таблицы '{main_table}'**\n\n"
        summary += f"📊 Проанализировано {len(statistical_results)} числовых переменных\n\n"

        if statistical_results:
            # Находим наиболее интересные статистики
            high_variance_cols = [r for r in statistical_results if r['coefficient_variation'] > 1]
            skewed_cols = [r for r in statistical_results if abs(r['skewness']) > 1]

            if high_variance_cols:
                summary += f"📊 Высокая вариативность найдена в {len(high_variance_cols)} переменных\n"
            if skewed_cols:
                summary += f"📈 Асимметричное распределение в {len(skewed_cols)} переменных\n"

            summary += f"\n🤖 **GPT Статистические Инсайты:**\n{gpt_insights}"

        return {
            'question': question,
            'data': clean_dataframe_for_json(stats_df),
            'summary': summary,
            'analyzed_tables': [main_table],
            'gpt_insights': gpt_insights,
            'statistical_insights': statistical_results,
            'correlation_matrix': correlation_matrix,
            'statistical_stats': {
                'variables_analyzed': len(statistical_results),
                'high_variance_count': len([r for r in statistical_results if r.get('coefficient_variation', 0) > 1]),
                'outliers_detected': sum(r.get('outliers_count', 0) for r in statistical_results)
            },
            'chart_data': self._prepare_chart_data_safe(stats_df, 'scatter', 'mean',
                                                        'std') if not stats_df.empty else None
        }

    def _classify_distribution(self, skewness: float, kurtosis: float) -> str:
        """Классифицирует тип распределения"""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return 'нормальное'
        elif skewness > 0.5:
            return 'правосторонняя асимметрия'
        elif skewness < -0.5:
            return 'левосторонняя асимметрия'
        elif kurtosis > 0.5:
            return 'высокий эксцесс'
        elif kurtosis < -0.5:
            return 'низкий эксцесс'
        else:
            return 'смешанное'

    def _detect_outliers_iqr(self, data: pd.Series) -> List:
        """Обнаружение выбросов методом IQR"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data < lower_bound) | (data > upper_bound)].tolist()

    def _analyze_predictive_patterns(self, question: str) -> Dict[str, Any]:
        """Анализ паттернов для предсказательной аналитики"""
        tables_mentioned = self._extract_mentioned_tables(question)
        if not tables_mentioned:
            tables_mentioned = list(self.df_manager.tables.keys())

        predictive_insights = []

        for table_name in tables_mentioned:
            df = self.df_manager.tables[table_name]

            # Поиск временных рядов
            date_cols = self._find_date_columns(df)
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if date_cols and len(numeric_cols) > 0:
                date_col = date_cols[0]
                try:
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                    df_clean = df_temp.dropna(subset=[date_col])

                    if len(df_clean) > 10:
                        # Анализ трендов по периодам
                        df_clean['period'] = df_clean[date_col].dt.to_period('M')

                        for num_col in numeric_cols[:3]:
                            monthly_data = df_clean.groupby('period')[num_col].agg(
                                ['count', 'sum', 'mean']).reset_index()

                            if len(monthly_data) > 3:
                                # Анализ тренда
                                trend_data = monthly_data['mean'].values
                                trend_direction = 'растущий' if len(trend_data) > 1 and trend_data[-1] > trend_data[
                                    0] else 'убывающий'

                                # Простая оценка сезонности
                                seasonality_score = float(np.std(trend_data) / np.mean(trend_data)) if np.mean(
                                    trend_data) > 0 else 0

                                # Прогнозируемость
                                predictability = self._assess_predictability(trend_data)

                                predictive_insights.append({
                                    'table': table_name,
                                    'metric': num_col,
                                    'trend_direction': trend_direction,
                                    'seasonality_score': round(seasonality_score, 3),
                                    'periods_analyzed': len(monthly_data),
                                    'predictability': predictability,
                                    'data_points': len(df_clean),
                                    'ml_readiness': self._assess_ml_readiness(df_clean, num_col)
                                })

                except Exception as e:
                    logger.error(f"Ошибка предиктивного анализа в {table_name}: {e}")

        predictive_df = pd.DataFrame(predictive_insights)

        # GPT анализ предиктивного потенциала
        try:
            gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                df=predictive_df,
                table_name='predictive_analysis',
                analysis_type='predictive_analysis',
                context={
                    'question': question,
                    'predictive_insights': predictive_insights[:3]
                }
            ) if not predictive_df.empty else {'gpt_analysis': ''}
            gpt_insights = gpt_analysis.get('gpt_analysis', '')
        except Exception as e:
            logger.error(f"Ошибка GPT предиктивного анализа: {e}")
            gpt_insights = 'GPT анализ предиктивного потенциала недоступен'

        if not predictive_df.empty:
            high_predictability = len(predictive_df[predictive_df['predictability'] == 'высокая'])
            ml_ready = len(predictive_df[predictive_df['ml_readiness'] >= 7])

            summary = f"🔮 **Анализ предсказательных паттернов: найдено {len(predictive_insights)} временных рядов**\n\n"
            summary += f"📊 **Статистика предиктивности:**\n"
            summary += f"• Высокая предсказуемость: {high_predictability} метрик\n"
            summary += f"• Готовы для ML: {ml_ready} метрик\n"
            summary += f"• Средняя сезонность: {predictive_df['seasonality_score'].mean():.3f}\n\n"
            summary += f"🤖 **GPT Прогнозы:**\n{gpt_insights}"
        else:
            summary = "🔮 Временные данные для предсказательного анализа не найдены"

        return {
            'question': question,
            'data': clean_dataframe_for_json(predictive_df),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'gpt_insights': gpt_insights,
            'predictive_patterns': predictive_insights,
            'predictive_stats': {
                'time_series_found': len(predictive_insights),
                'high_predictability_count': len([p for p in predictive_insights if p['predictability'] == 'высокая']),
                'ml_ready_count': len([p for p in predictive_insights if p.get('ml_readiness', 0) >= 7])
            },
            'chart_data': self._prepare_chart_data_safe(predictive_df, 'scatter', 'seasonality_score',
                                                        'data_points') if not predictive_df.empty else None
        }

    def _assess_predictability(self, trend_data: np.ndarray) -> str:
        """Оценивает предсказуемость временного ряда"""
        if len(trend_data) < 3:
            return 'низкая'

        try:
            # Коэффициент вариации
            cv = np.std(trend_data) / np.mean(trend_data) if np.mean(trend_data) > 0 else float('inf')

            if cv < 0.3:
                return 'высокая'
            elif cv < 0.6:
                return 'средняя'
            else:
                return 'низкая'
        except:
            return 'низкая'

    def _assess_ml_readiness(self, df: pd.DataFrame, target_col: str) -> int:
        """Оценивает готовность данных для машинного обучения (0-10)"""
        score = 0

        try:
            # Размер данных
            if len(df) > 1000:
                score += 3
            elif len(df) > 100:
                score += 2
            elif len(df) > 50:
                score += 1

            # Количество признаков
            feature_count = len(df.select_dtypes(include=[np.number]).columns) - 1  # -1 для целевой переменной
            if feature_count > 10:
                score += 2
            elif feature_count > 5:
                score += 1

            # Качество данных
            missing_pct = df[target_col].isnull().sum() / len(df) * 100
            if missing_pct < 5:
                score += 2
            elif missing_pct < 15:
                score += 1

            # Наличие категориальных признаков
            categorical_count = len(df.select_dtypes(include=['object']).columns)
            if 0 < categorical_count < 10:
                score += 1

            # Временная составляющая
            date_cols = self._find_date_columns(df)
            if date_cols:
                score += 1

        except Exception as e:
            logger.warning(f"Ошибка оценки ML готовности: {e}")

        return min(10, score)

    def _analyze_general(self, question: str) -> Dict[str, Any]:
        """Общий анализ для неопределенных вопросов с GPT помощью"""
        tables_mentioned = self._extract_mentioned_tables(question)
        if not tables_mentioned:
            if self.df_manager.tables:
                largest_table = max(self.df_manager.tables.items(), key=lambda x: len(x[1]))
                tables_mentioned = [largest_table[0]]
            else:
                return {
                    'question': question,
                    'data': [],
                    'summary': 'Нет доступных таблиц для анализа',
                    'analyzed_tables': []
                }

        table_name = tables_mentioned[0]
        df = self.df_manager.tables[table_name]

        # Создаем общую статистику
        general_stats = {
            'table_name': table_name,
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns)),
            'columns_list': list(df.columns),
            'data_types': convert_to_serializable(df.dtypes.value_counts().to_dict()),
            'missing_values': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum()),
            'memory_usage_mb': round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 2)
        }

        # Дополнительная аналитика
        numeric_summary = {}
        categorical_summary = {}

        # Числовые колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:5]:  # Первые 5 числовых колонок
                try:
                    numeric_summary[col] = {
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'count': int(df[col].count())
                    }
                except Exception as e:
                    logger.warning(f"Ошибка анализа числовой колонки {col}: {e}")
                    numeric_summary[col] = {'error': str(e)}

                # Категориальные колонки
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols[:5]:  # Первые 5 категориальных колонок
                try:
                    value_counts = df[col].value_counts()
                    if len(value_counts) > 0:
                        categorical_summary[col] = {
                            'unique_values': int(df[col].nunique()),
                            'most_common': str(value_counts.index[0]),
                            'most_common_count': int(value_counts.iloc[0]),
                            'top_5_values': convert_to_serializable(value_counts.head(5).to_dict())
                        }
                except Exception as e:
                    logger.warning(f"Ошибка анализа категориальной колонки {col}: {e}")
                    categorical_summary[col] = {'error': str(e)}

            # GPT анализ общих данных
            try:
                gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
                    df=df,
                    table_name=table_name,
                    analysis_type="business_insights",
                    context={
                        'question': question,
                        'general_stats': general_stats,
                        'numeric_summary': numeric_summary,
                        'categorical_summary': categorical_summary
                    }
                )
                gpt_insights = gpt_analysis.get('gpt_analysis', 'GPT анализ недоступен')
            except Exception as e:
                logger.error(f"Ошибка GPT анализа: {e}")
                gpt_insights = 'GPT анализ недоступен'

            # Добавляем примеры данных
            sample_data = clean_dataframe_for_json(df.head(10))

            summary = f"📊 **Общий анализ таблицы '{table_name}'**\n\n"
            summary += f"📈 Базовая статистика: {len(df):,} записей, {len(df.columns)} колонок\n"
            summary += f"❌ Пропущенных значений: {general_stats['missing_values']}\n"
            summary += f"🔄 Дубликатов: {general_stats['duplicate_rows']}\n\n"
            summary += f"🤖 **GPT Инсайты:**\n{gpt_insights}"

            return {
                'question': question,
                'data': sample_data,
                'summary': summary,
                'analyzed_tables': [table_name],
                'gpt_insights': gpt_insights,
                'general_stats': general_stats,
                'numeric_summary': numeric_summary,
                'categorical_summary': categorical_summary,
                'chart_data': self._prepare_chart_data_safe(df.head(100), 'histogram', df.columns[0], None)
            }

        # =============== ДОПОЛНИТЕЛЬНЫЕ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ===============

        def _compare_columns_in_table(self, table_name: str, question: str) -> Dict[str, Any]:
            """Сравнение колонок внутри одной таблицы"""
            df = self.df_manager.tables[table_name]

            # Получаем числовые колонки для сравнения
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) < 2:
                return {
                    'question': question,
                    'data': [],
                    'summary': f'В таблице {table_name} недостаточно числовых колонок для сравнения',
                    'analyzed_tables': [table_name]
                }

            comparison_data = []

            # Сравниваем первые несколько числовых колонок
            for col in numeric_cols[:5]:
                try:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        comparison_data.append({
                            'column': col,
                            'mean': round(float(col_data.mean()), 2),
                            'median': round(float(col_data.median()), 2),
                            'std': round(float(col_data.std()), 2),
                            'min': float(col_data.min()),
                            'max': float(col_data.max()),
                            'range': float(col_data.max() - col_data.min()),
                            'count': int(len(col_data))
                        })
                except Exception as e:
                    logger.error(f"Ошибка сравнения колонки {col}: {e}")

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)

                # Находим колонку с максимальным средним значением
                max_mean_col = comparison_df.loc[comparison_df['mean'].idxmax(), 'column']
                min_mean_col = comparison_df.loc[comparison_df['mean'].idxmin(), 'column']

                summary = f"📊 Сравнение числовых колонок в таблице '{table_name}'. "
                summary += f"Максимальное среднее значение: {max_mean_col}. "
                summary += f"Минимальное среднее значение: {min_mean_col}."
            else:
                comparison_df = pd.DataFrame()
                summary = "Не удалось сравнить колонки"

            return {
                'question': question,
                'data': clean_dataframe_for_json(comparison_df),
                'summary': summary,
                'analyzed_tables': [table_name],
                'chart_data': self._prepare_chart_data_safe(comparison_df, 'bar', 'column', 'mean')
            }

        # =============== ДОПОЛНИТЕЛЬНЫЕ АНАЛИЗЫ ===============

        logger.info("DataFrame Analyzer полностью загружен с расширенной функциональностью")

    def _analyze_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ категориальных колонок"""
        categorical_analysis = {}
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            try:
                value_counts = df[col].value_counts()
                categorical_analysis[col] = {
                    'unique_count': int(len(value_counts)),
                    'top_values': convert_to_serializable(value_counts.head(5).to_dict()),
                    'null_count': int(df[col].isnull().sum()),
                    'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else 'N/A',
                    'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'diversity_score': round(len(value_counts) / len(df) * 100, 2),  # Процент уникальности
                    'fill_rate': round((1 - df[col].isnull().sum() / len(df)) * 100, 2)
                }

                # Дополнительный анализ для потенциальных ID колонок
                if 'id' in col.lower():
                    categorical_analysis[col]['is_likely_id'] = df[col].nunique() / len(df) > 0.8

            except Exception as e:
                logger.error(f"Ошибка анализа категориальной колонки {col}: {e}")
                categorical_analysis[col] = {
                    'unique_count': 0,
                    'top_values': {},
                    'null_count': int(df[col].isnull().sum()),
                    'most_common': 'Error',
                    'most_common_count': 0,
                    'error': str(e)
                }

        return categorical_analysis

    def _detect_anomalies_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение аномалий в DataFrame"""
        anomalies = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            try:
                if df[col].count() > 10:  # Минимум данных для анализа
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    if IQR > 0:  # Избегаем деления на 0
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

                        if len(outliers) > 0:
                            sample_values = outliers[col].head(5).tolist()
                            anomalies.append({
                                'type': 'outlier',
                                'column': col,
                                'count': int(len(outliers)),
                                'percentage': round(float(len(outliers) / len(df) * 100), 2),
                                'lower_bound': round(float(lower_bound), 2),
                                'upper_bound': round(float(upper_bound), 2),
                                'sample_values': [float(v) for v in sample_values],
                                'severity': 'high' if len(outliers) / len(df) > 0.1 else 'medium',
                                'description': f'Найдено {len(outliers)} выбросов в колонке {col}'
                            })

                    # Дополнительные проверки аномалий
                    # Проверка на подозрительно частые значения
                    if df[col].dtype in ['int64', 'float64']:
                        value_counts = df[col].value_counts()
                        if len(value_counts) > 0:
                            most_frequent_count = value_counts.iloc[0]
                            if most_frequent_count / len(df) > 0.5:  # Больше 50% одинаковых значений
                                anomalies.append({
                                    'type': 'frequent_value',
                                    'column': col,
                                    'count': int(most_frequent_count),
                                    'percentage': round(float(most_frequent_count / len(df) * 100), 2),
                                    'value': float(value_counts.index[0]),
                                    'severity': 'medium',
                                    'description': f'Значение {value_counts.index[0]} встречается в {most_frequent_count} из {len(df)} записей'
                                })

                    # Проверка на отрицательные значения в колонках, где они не ожидаются
                    if any(keyword in col.lower() for keyword in ['price', 'cost', 'amount', 'quantity', 'age']):
                        negative_count = (df[col] < 0).sum()
                        if negative_count > 0:
                            anomalies.append({
                                'type': 'negative_values',
                                'column': col,
                                'count': int(negative_count),
                                'percentage': round(float(negative_count / len(df) * 100), 2),
                                'severity': 'high',
                                'description': f'Найдено {negative_count} отрицательных значений в колонке {col}, где они не ожидаются'
                            })

            except Exception as e:
                logger.error(f"Ошибка обнаружения аномалий в колонке {col}: {e}")

        # Проверки для категориальных данных
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            try:
                # Проверка на подозрительно много уникальных значений
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.9:  # Больше 90% уникальных значений
                    anomalies.append({
                        'type': 'high_cardinality',
                        'column': col,
                        'count': int(df[col].nunique()),
                        'percentage': round(float(unique_ratio * 100), 2),
                        'severity': 'low',
                        'description': f'Колонка {col} имеет подозрительно много уникальных значений ({df[col].nunique()} из {len(df)})'
                    })

            except Exception as e:
                logger.error(f"Ошибка анализа категориальных аномалий в колонке {col}: {e}")

        return anomalies

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ пропущенных значений"""
        try:
            missing_data = df.isnull().sum()
            total_missing = missing_data.sum()
            total_cells = len(df) * len(df.columns)

            missing_analysis = {
                'total_missing': int(total_missing),
                'missing_percentage': round(float(total_missing / total_cells * 100), 2),
                'columns_with_missing': {},
                'missing_patterns': {},
                'data_quality_score': 0
            }

            # Анализ по колонкам
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    missing_percentage = missing_count / len(df) * 100
                    missing_analysis['columns_with_missing'][col] = {
                        'count': int(missing_count),
                        'percentage': round(float(missing_percentage), 2),
                        'severity': 'critical' if missing_percentage > 50 else 'high' if missing_percentage > 20 else 'medium'
                    }

            # Паттерны пропусков
            if len(missing_analysis['columns_with_missing']) > 1:
                # Ищем колонки с похожими паттернами пропусков
                missing_patterns = {}
                for col1, info1 in missing_analysis['columns_with_missing'].items():
                    for col2, info2 in missing_analysis['columns_with_missing'].items():
                        if col1 != col2:
                            diff = abs(info1['percentage'] - info2['percentage'])
                            if diff < 5:  # Разница меньше 5%
                                pattern_key = f"{col1}_similar_to_{col2}"
                                missing_patterns[pattern_key] = {
                                    'columns': [col1, col2],
                                    'similarity': round(100 - diff, 1)
                                }

                missing_analysis['missing_patterns'] = missing_patterns

            # Общий балл качества данных (по пропускам)
            completeness = (1 - total_missing / total_cells) * 100
            missing_analysis['data_quality_score'] = round(completeness, 1)

            return missing_analysis

        except Exception as e:
            logger.error(f"Ошибка анализа пропущенных значений: {e}")
            return {
                'total_missing': 0,
                'missing_percentage': 0,
                'columns_with_missing': {},
                'error': str(e)
            }

    def _analyze_correlations_single_table(self, df: pd.DataFrame, table_name: str = None) -> List[Dict[str, Any]]:
        """Анализ корреляций в одной таблице"""
        correlations = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()

                # Извлекаем значимые корреляции
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]

                        if not pd.isna(corr_value) and abs(corr_value) > 0.3:  # Значимая корреляция
                            col1 = corr_matrix.columns[i]
                            col2 = corr_matrix.columns[j]

                            # Определяем силу корреляции
                            abs_corr = abs(corr_value)
                            if abs_corr > 0.8:
                                strength = 'очень сильная'
                            elif abs_corr > 0.6:
                                strength = 'сильная'
                            elif abs_corr > 0.4:
                                strength = 'умеренная'
                            else:
                                strength = 'слабая'

                            correlations.append({
                                'table': table_name or 'unknown',
                                'column1': col1,
                                'column2': col2,
                                'correlation': round(float(corr_value), 3),
                                'correlation_abs': round(float(abs_corr), 3),
                                'strength': strength,
                                'direction': 'положительная' if corr_value > 0 else 'отрицательная',
                                'business_meaning': self._interpret_correlation(col1, col2, corr_value),
                                'statistical_significance': 'высокая' if abs_corr > 0.7 else 'умеренная'
                            })

                # Сортируем по силе корреляции
                correlations.sort(key=lambda x: x['correlation_abs'], reverse=True)

            except Exception as e:
                logger.error(f"Ошибка анализа корреляций: {e}")

        return correlations

    def _interpret_correlation(self, col1: str, col2: str, correlation: float) -> str:
        """Интерпретирует бизнес-смысл корреляции"""
        try:
            col1_lower = col1.lower()
            col2_lower = col2.lower()

            # Финансовые корреляции
            if any(word in col1_lower for word in ['price', 'cost']) and any(
                    word in col2_lower for word in ['revenue', 'sales']):
                return "Связь между ценообразованием и доходами"
            elif any(word in col1_lower for word in ['quantity', 'amount']) and any(
                    word in col2_lower for word in ['revenue', 'total']):
                return "Объемно-стоимостная зависимость"

            # Временные корреляции
            elif any(word in col1_lower for word in ['time', 'date', 'age']) and any(
                    word in col2_lower for word in ['value', 'score']):
                return "Временная динамика показателя"

            # Качественные корреляции
            elif any(word in col1_lower for word in ['quality', 'rating']) and any(
                    word in col2_lower for word in ['price', 'cost']):
                return "Зависимость качества и стоимости"

            # Общая интерпретация
            else:
                if correlation > 0:
                    return f"При росте {col1} увеличивается {col2}"
                else:
                    return f"При росте {col1} уменьшается {col2}"

        except Exception:
            return "Требует дополнительного анализа"

    def _prepare_business_metrics_chart(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Подготовка графика для бизнес-метрик"""
        try:
            # Извлекаем числовые метрики для визуализации
            chart_metrics = []

            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not key.endswith('_count') and key not in ['total_records']:
                    # Форматируем название метрики
                    formatted_name = key.replace('_', ' ').title()

                    # Добавляем единицы измерения где возможно
                    if 'percentage' in key.lower() or 'rate' in key.lower():
                        formatted_name += ' (%)'
                    elif 'total' in key.lower() or 'sum' in key.lower():
                        formatted_name += ' (Total)'
                    elif 'average' in key.lower() or 'avg' in key.lower():
                        formatted_name += ' (Avg)'

                    chart_metrics.append({
                        'metric': formatted_name,
                        'value': float(value),
                        'category': self._categorize_metric(key)
                    })

            if len(chart_metrics) > 0:
                # Создаем DataFrame для графика
                chart_df = pd.DataFrame(chart_metrics)

                # Сортируем по значению для лучшей визуализации
                chart_df = chart_df.sort_values('value', ascending=False)

                return self._prepare_chart_data_safe(chart_df, 'bar', 'metric', 'value')

            return None

        except Exception as e:
            logger.error(f"Ошибка подготовки business metrics chart: {e}")
            return None

    def _categorize_metric(self, metric_name: str) -> str:
        """Категоризирует метрику для лучшей группировки"""
        metric_lower = metric_name.lower()

        if any(word in metric_lower for word in ['revenue', 'sales', 'amount', 'total', 'sum']):
            return 'financial'
        elif any(word in metric_lower for word in ['percentage', 'rate', 'ratio']):
            return 'percentage'
        elif any(word in metric_lower for word in ['average', 'mean', 'median']):
            return 'average'
        elif any(word in metric_lower for word in ['count', 'number', 'quantity']):
            return 'count'
        elif any(word in metric_lower for word in ['completeness', 'quality', 'score']):
            return 'quality'
        else:
            return 'other'
