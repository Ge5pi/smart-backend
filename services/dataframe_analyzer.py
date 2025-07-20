# services/dataframe_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from services.dataframe_manager import DataFrameManager
from datetime import datetime
import re
import sys
import os

# Добавляем путь для импорта utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_serializer import convert_to_serializable, clean_dataframe_for_json

logger = logging.getLogger(__name__)


class DataFrameAnalyzer:
    """Аналитический движок для работы с DataFrame"""

    def __init__(self, df_manager: DataFrameManager):
        self.df_manager = df_manager
        self.analysis_cache = {}

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

    def _analyze_overview(self) -> Dict[str, Any]:
        """Общий обзор данных"""
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
                'memory_usage_mb': round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 2)
            })

        overview_df = pd.DataFrame(overview_data)

        # Информация о связях
        relations_info = []
        for rel in self.df_manager.relations:
            relations_info.append(f"{rel.from_table}.{rel.from_column} -> {rel.to_table}.{rel.to_column}")

        total_rows = overview_df['row_count'].sum() if not overview_df.empty else 0
        total_tables = len(self.df_manager.tables)
        relations_count = len(self.df_manager.relations)
        total_memory = overview_df['memory_usage_mb'].sum() if not overview_df.empty else 0

        summary = (f"База данных содержит {total_tables} таблиц с {total_rows} записями. "
                   f"Обнаружено {relations_count} связей между таблицами. "
                   f"Общий объем данных в памяти: {total_memory:.2f} MB.")

        return {
            'question': 'Общий обзор базы данных',
            'data': clean_dataframe_for_json(overview_df),
            'summary': summary,
            'analyzed_tables': list(self.df_manager.tables.keys()),
            'relations': relations_info,
            'chart_data': self._prepare_chart_data_safe(overview_df, 'bar', 'table_name', 'row_count')
        }

    def _analyze_single_table(self, table_name: str) -> Dict[str, Any]:
        """Детальный анализ одной таблицы"""
        if table_name not in self.df_manager.tables:
            # Пытаемся найти таблицу по частичному совпадению
            matching_tables = [t for t in self.df_manager.tables.keys()
                               if table_name.lower() in t.lower()]
            if matching_tables:
                table_name = matching_tables[0]
            else:
                return {
                    'question': f'Анализ таблицы {table_name}',
                    'error': f'Таблица {table_name} не найдена',
                    'data': [],
                    'summary': f'Таблица {table_name} не найдена в загруженных данных',
                    'analyzed_tables': []
                }

        df = self.df_manager.tables[table_name]

        # Базовая информация
        basic_info = {
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'memory_mb': round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 3)
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

        # Создаем сводку
        summary_parts = [
            f"Таблица '{table_name}' содержит {len(df)} записей и {len(df.columns)} колонок.",
            f"Числовых колонок: {len(numeric_cols)}, категориальных: {len(df.select_dtypes(include=['object']).columns)}."
        ]

        if anomalies:
            summary_parts.append(f"Обнаружено {len(anomalies)} типов аномалий.")

        if correlations:
            summary_parts.append(f"Найдено {len(correlations)} значимых корреляций.")

        if missing_analysis['total_missing'] > 0:
            summary_parts.append(
                f"Пропущенных значений: {missing_analysis['total_missing']} ({missing_analysis['missing_percentage']:.1f}%).")

        return {
            'question': f'Детальный анализ таблицы {table_name}',
            'data': clean_dataframe_for_json(df.head(20)),
            'summary': ' '.join(summary_parts),
            'analyzed_tables': [table_name],
            'basic_info': basic_info,
            'numeric_stats': numeric_stats,
            'categorical_stats': categorical_stats,
            'anomalies': anomalies,
            'correlations': correlations,
            'missing_analysis': missing_analysis,
            'chart_data': self._create_table_chart(df, table_name)
        }

    def _analyze_relationships(self) -> Dict[str, Any]:
        """Анализ связей между таблицами"""
        if not self.df_manager.relations:
            return {
                'question': 'Анализ связей между таблицами',
                'data': [],
                'summary': 'Связи между таблицами не обнаружены',
                'analyzed_tables': []
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
                    'relation_type': relation.relation_type
                })

            except Exception as e:
                logger.error(f"Ошибка анализа связи {relation.from_table}->{relation.to_table}: {e}")

        if not relationship_data:
            return {
                'question': 'Анализ связей между таблицами',
                'data': [],
                'summary': 'Не удалось проанализировать связи',
                'analyzed_tables': []
            }

        relationships_df = pd.DataFrame(relationship_data)

        # Находим самые сильные и слабые связи
        if not relationships_df.empty:
            strongest = relationships_df.loc[relationships_df['relationship_strength'].idxmax()]
            weakest = relationships_df.loc[relationships_df['relationship_strength'].idxmin()]
            avg_strength = relationships_df['relationship_strength'].mean()

            summary = (f"Найдено {len(self.df_manager.relations)} связей между таблицами. "
                       f"Средняя сила связей: {avg_strength:.1f}%. "
                       f"Самая сильная: {strongest['from_table']}->{strongest['to_table']} ({strongest['relationship_strength']:.1f}%). "
                       f"Самая слабая: {weakest['from_table']}->{weakest['to_table']} ({weakest['relationship_strength']:.1f}%).")
        else:
            summary = "Анализ связей не дал результатов"

        analyzed_tables = list(set([r['from_table'] for r in relationship_data] +
                                   [r['to_table'] for r in relationship_data]))

        return {
            'question': 'Анализ связей между таблицами',
            'data': clean_dataframe_for_json(relationships_df),
            'summary': summary,
            'analyzed_tables': analyzed_tables,
            'chart_data': self._prepare_chart_data_safe(relationships_df, 'bar',
                                                        'from_table', 'relationship_strength')
        }

    def _analyze_aggregations(self, question: str) -> Dict[str, Any]:
        """Анализ с агрегациями"""
        tables_mentioned = self._extract_mentioned_tables(question)

        if not tables_mentioned:
            # Берем самую большую таблицу
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
                            if pd.isna(value):
                                value = 0
                            col_data[func] = float(value)
                        except:
                            col_data[func] = 0

                    aggregated_data.append(col_data)

                results.append({
                    'table': table_name,
                    'type': 'numeric_aggregation',
                    'data': convert_to_serializable(agg_result)
                })

            # Категориальные агрегации
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols[:3]:  # Ограничиваем количество
                    try:
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
                                'data': convert_to_serializable(value_counts)
                            })
                    except Exception as e:
                        logger.error(f"Ошибка агрегации категориальной колонки {col}: {e}")

        # Создаем итоговый DataFrame
        if aggregated_data:
            main_result = pd.DataFrame(aggregated_data)
            summary = f"Агрегированный анализ {len(tables_mentioned)} таблиц. Проанализировано {len(results)} различных метрик."
        else:
            main_result = pd.DataFrame()
            summary = "Не удалось выполнить агрегацию данных"

        return {
            'question': question,
            'data': clean_dataframe_for_json(main_result),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'detailed_results': convert_to_serializable(results),
            'chart_data': self._create_aggregation_chart(main_result)
        }

    def _analyze_trends(self, question: str) -> Dict[str, Any]:
        """Временной анализ"""
        trend_results = []

        for table_name, df in self.df_manager.tables.items():
            # Ищем колонки с датами
            date_cols = self._find_date_columns(df)

            if date_cols:
                date_col = date_cols[0]
                try:
                    df_copy = df.copy()
                    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')

                    # Убираем строки с некорректными датами
                    df_clean = df_copy.dropna(subset=[date_col])

                    if len(df_clean) > 0:
                        # Группируем по периодам (месяцам)
                        df_clean['period'] = df_clean[date_col].dt.to_period('M')
                        trend_data = df_clean.groupby('period').size().reset_index(name='count')
                        trend_data['period_str'] = trend_data['period'].astype(str)

                        # Вычисляем статистику тренда
                        if len(trend_data) > 1:
                            # Простой анализ роста/падения
                            first_val = int(trend_data['count'].iloc[0])
                            last_val = int(trend_data['count'].iloc[-1])
                            trend_direction = "рост" if last_val > first_val else "падение"
                            trend_percent = abs((last_val - first_val) / first_val * 100) if first_val > 0 else 0
                        else:
                            trend_direction = "недостаточно данных"
                            trend_percent = 0

                        # Конвертируем timestamp в строки
                        min_date = df_clean[date_col].min()
                        max_date = df_clean[date_col].max()
                        try:
                            date_range = f"{min_date.strftime('%Y-%m-%d')} - {max_date.strftime('%Y-%m-%d')}"
                        except:
                            date_range = f"{str(min_date)[:10]} - {str(max_date)[:10]}"

                        trend_results.append({
                            'table': table_name,
                            'date_column': date_col,
                            'trend_data': clean_dataframe_for_json(trend_data),
                            'total_records': int(len(df_clean)),
                            'periods_count': int(len(trend_data)),
                            'trend_direction': trend_direction,
                            'trend_percent': round(float(trend_percent), 1),
                            'date_range': date_range
                        })

                except Exception as e:
                    logger.error(f"Ошибка анализа трендов для {table_name}: {e}")

        if trend_results:
            # Берем самый информативный тренд
            main_trend = max(trend_results, key=lambda x: x['total_records'])
            summary = (f"Временной анализ найден в {len(trend_results)} таблицах. "
                       f"Основной тренд в таблице '{main_trend['table']}' по колонке '{main_trend['date_column']}': "
                       f"{main_trend['trend_direction']} на {main_trend['trend_percent']:.1f}% за период {main_trend['date_range']}.")

            main_data = main_trend['trend_data']
        else:
            main_data = []
            summary = "Временные данные для анализа трендов не найдены"

        analyzed_tables = [r['table'] for r in trend_results]

        return {
            'question': question,
            'data': main_data,
            'summary': summary,
            'analyzed_tables': analyzed_tables,
            'all_trends': trend_results,
            'chart_data': self._prepare_chart_data_safe_from_list(main_data, 'line', 'period_str', 'count')
        }

    def _analyze_correlations(self, question: str) -> Dict[str, Any]:
        """Анализ корреляций"""
        tables_mentioned = self._extract_mentioned_tables(question)

        if not tables_mentioned:
            tables_mentioned = list(self.df_manager.tables.keys())

        all_correlations = []

        for table_name in tables_mentioned:
            df = self.df_manager.tables[table_name]
            correlations = self._analyze_correlations_single_table(df, table_name)
            all_correlations.extend(correlations)

        if all_correlations:
            # Создаем DataFrame с корреляциями
            corr_df = pd.DataFrame(all_correlations)

            # Сортируем по силе корреляции
            corr_df['abs_correlation'] = abs(corr_df['correlation'])
            corr_df = corr_df.sort_values('abs_correlation', ascending=False)
            corr_df = corr_df.drop('abs_correlation', axis=1)  # Убираем вспомогательную колонку

            # Находим самые сильные корреляции
            strong_correlations = corr_df[abs(corr_df['correlation']) > 0.7]

            summary = (f"Анализ корреляций в {len(tables_mentioned)} таблицах. "
                       f"Найдено {len(all_correlations)} корреляций, "
                       f"из них {len(strong_correlations)} сильных (>0.7).")

        else:
            corr_df = pd.DataFrame()
            summary = "Значимые корреляции не найдены"

        return {
            'question': question,
            'data': clean_dataframe_for_json(corr_df),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'chart_data': self._prepare_chart_data_safe(corr_df, 'scatter', 'column1', 'correlation')
        }

    def _analyze_comparison(self, question: str) -> Dict[str, Any]:
        """Анализ сравнения между таблицами"""
        tables_mentioned = self._extract_mentioned_tables(question)

        if len(tables_mentioned) < 2:
            # Если упомянута одна таблица, сравниваем колонки внутри неё
            if len(tables_mentioned) == 1:
                return self._compare_columns_in_table(tables_mentioned[0], question)
            else:
                # Берем две самые большие таблицы
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

        for table_name in tables_mentioned:
            df = self.df_manager.tables[table_name]

            # Базовая статистика таблицы
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            stats = {
                'table': table_name,
                'rows': int(len(df)),
                'columns': int(len(df.columns)),
                'numeric_columns': int(len(numeric_cols)),
                'categorical_columns': int(len(categorical_cols)),
                'null_values': int(df.isnull().sum().sum()),
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
                    stats.update({
                        'avg_mean': 0,
                        'avg_std': 0,
                        'avg_min': 0,
                        'avg_max': 0
                    })

            comparison_data.append(stats)

        comparison_df = pd.DataFrame(comparison_data)

        if len(comparison_data) >= 2:
            # Находим различия
            table1, table2 = comparison_data[0], comparison_data[1]
            differences = []

            for key in ['rows', 'columns', 'numeric_columns', 'categorical_columns']:
                if key in table1 and key in table2:
                    diff = abs(table1[key] - table2[key])
                    if diff > 0:
                        differences.append(f"{key}: {table1[key]} vs {table2[key]} (разница: {diff})")

            summary = (f"Сравнение таблиц {' vs '.join(tables_mentioned)}. "
                       f"Основные различия: {'; '.join(differences[:3])}.")
        else:
            summary = "Недостаточно данных для сравнения"

        return {
            'question': question,
            'data': clean_dataframe_for_json(comparison_df),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'chart_data': self._prepare_chart_data_safe(comparison_df, 'bar', 'table', 'rows')
        }

    def _analyze_anomalies(self, question: str) -> Dict[str, Any]:
        """Специальный анализ аномалий"""
        tables_mentioned = self._extract_mentioned_tables(question)

        if not tables_mentioned:
            tables_mentioned = list(self.df_manager.tables.keys())

        all_anomalies = []

        for table_name in tables_mentioned:
            df = self.df_manager.tables[table_name]
            anomalies = self._detect_anomalies_dataframe(df)

            for anomaly in anomalies:
                anomaly['table'] = table_name
                all_anomalies.append(anomaly)

        if all_anomalies:
            anomalies_df = pd.DataFrame(all_anomalies)
            total_anomalies = sum(a['count'] for a in all_anomalies)

            summary = (f"Обнаружено {len(all_anomalies)} типов аномалий в {len(tables_mentioned)} таблицах. "
                       f"Общее количество аномальных записей: {total_anomalies}.")
        else:
            anomalies_df = pd.DataFrame()
            summary = "Аномалии в данных не обнаружены"

        return {
            'question': question,
            'data': clean_dataframe_for_json(anomalies_df),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'chart_data': self._prepare_chart_data_safe(anomalies_df, 'bar', 'table', 'count')
        }

    def _analyze_general(self, question: str) -> Dict[str, Any]:
        """Общий анализ для неопределенных вопросов"""
        # Пытаемся извлечь упомянутые таблицы
        tables_mentioned = self._extract_mentioned_tables(question)

        if not tables_mentioned:
            # Берем самую большую таблицу
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

        # Выполняем базовый анализ первой упомянутой таблицы
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
            'duplicate_rows': int(df.duplicated().sum())
        }

        # Добавляем примеры данных
        sample_data = clean_dataframe_for_json(df.head(10))

        summary = (f"Общий анализ таблицы '{table_name}': {len(df)} записей, "
                   f"{len(df.columns)} колонок. "
                   f"Пропущенных значений: {general_stats['missing_values']}, "
                   f"дубликатов: {general_stats['duplicate_rows']}.")

        return {
            'question': question,
            'data': sample_data,
            'summary': summary,
            'analyzed_tables': [table_name],
            'general_stats': general_stats,
            'chart_data': None
        }

    # Вспомогательные методы
    def _categorize_question(self, question: str) -> str:
        """Категоризирует тип вопроса"""
        question_lower = question.lower()

        if any(keyword in question_lower for keyword in ['обзор', 'общий', 'статистика', 'overview', 'структура']):
            return 'overview'
        elif any(keyword in question_lower for keyword in ['таблица', 'table', 'анализ таблицы']):
            return 'table_analysis'
        elif any(keyword in question_lower for keyword in ['связь', 'связи', 'отношения', 'relation']):
            return 'relationship_analysis'
        elif any(keyword in question_lower for keyword in
                 ['сумма', 'среднее', 'количество', 'агрегация', 'группировка']):
            return 'aggregation'
        elif any(keyword in question_lower for keyword in ['тренд', 'динамика', 'время', 'временной', 'дата']):
            return 'trend_analysis'
        elif any(keyword in question_lower for keyword in ['корреляция', 'зависимость', 'связанность']):
            return 'correlation'
        elif any(keyword in question_lower for keyword in ['сравнение', 'сравнить', 'различия', 'vs']):
            return 'comparison'
        elif any(keyword in question_lower for keyword in ['аномалии', 'выбросы', 'anomaly', 'outlier']):
            return 'anomalies'
        else:
            return 'general'

    def _extract_table_name(self, question: str) -> str:
        """Извлекает название таблицы из вопроса"""
        question_lower = question.lower()

        # Прямое совпадение
        for table_name in self.df_manager.tables.keys():
            if table_name.lower() in question_lower:
                return table_name

        # Частичное совпадение
        for table_name in self.df_manager.tables.keys():
            table_words = table_name.lower().split('_')
            if any(word in question_lower for word in table_words if len(word) > 3):
                return table_name

        # Возвращаем самую большую таблицу как fallback
        if self.df_manager.tables:
            return max(self.df_manager.tables.items(), key=lambda x: len(x[1]))[0]
        else:
            return "unknown"

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
                    'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                }
            except Exception as e:
                logger.error(f"Ошибка анализа категориальной колонки {col}: {e}")
                categorical_analysis[col] = {
                    'unique_count': 0,
                    'top_values': {},
                    'null_count': int(df[col].isnull().sum()),
                    'most_common': 'Error',
                    'most_common_count': 0
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
                                'column': col,
                                'count': int(len(outliers)),
                                'percentage': round(float(len(outliers) / len(df) * 100), 2),
                                'lower_bound': round(float(lower_bound), 2),
                                'upper_bound': round(float(upper_bound), 2),
                                'sample_values': [float(v) for v in sample_values]
                            })
            except Exception as e:
                logger.error(f"Ошибка обнаружения аномалий в колонке {col}: {e}")

        return anomalies

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ пропущенных значений"""
        try:
            missing_data = df.isnull().sum()
            total_missing = missing_data.sum()

            missing_analysis = {
                'total_missing': int(total_missing),
                'missing_percentage': round(float(total_missing / (len(df) * len(df.columns)) * 100), 2),
                'columns_with_missing': {}
            }

            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    missing_analysis['columns_with_missing'][col] = {
                        'count': int(missing_count),
                        'percentage': round(float(missing_count / len(df) * 100), 2)
                    }

            return missing_analysis
        except Exception as e:
            logger.error(f"Ошибка анализа пропущенных значений: {e}")
            return {
                'total_missing': 0,
                'missing_percentage': 0,
                'columns_with_missing': {}
            }

    def _analyze_correlations_single_table(self, df: pd.DataFrame, table_name: str = None) -> List[Dict[str, Any]]:
        """Анализ корреляций в одной таблице"""
        correlations = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()

                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_value) and abs(corr_value) > 0.5:  # Значимая корреляция
                            correlations.append({
                                'table': table_name or 'unknown',
                                'column1': corr_matrix.columns[i],
                                'column2': corr_matrix.columns[j],
                                'correlation': round(float(corr_value), 3),
                                'strength': 'сильная' if abs(corr_value) > 0.7 else 'умеренная',
                                'direction': 'положительная' if corr_value > 0 else 'отрицательная'
                            })
            except Exception as e:
                logger.error(f"Ошибка анализа корреляций: {e}")

        return correlations

    def _compare_columns_in_table(self, table_name: str, question: str) -> Dict[str, Any]:
        """Сравнение колонок внутри одной таблицы"""
        df = self.df_manager.tables[table_name]

        # Анализируем числовые колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        comparison_data = []

        for col in numeric_cols:
            try:
                col_stats = {
                    'column': col,
                    'count': int(df[col].count()),
                    'mean': round(float(df[col].mean()), 2),
                    'median': round(float(df[col].median()), 2),
                    'std': round(float(df[col].std()), 2),
                    'min': round(float(df[col].min()), 2),
                    'max': round(float(df[col].max()), 2),
                    'range': round(float(df[col].max() - df[col].min()), 2)
                }
                comparison_data.append(col_stats)
            except Exception as e:
                logger.error(f"Ошибка анализа колонки {col}: {e}")

        comparison_df = pd.DataFrame(comparison_data)

        if not comparison_df.empty:
            # Находим колонку с наибольшими и наименьшими значениями
            max_mean_col = comparison_df.loc[comparison_df['mean'].idxmax(), 'column']
            min_mean_col = comparison_df.loc[comparison_df['mean'].idxmin(), 'column']

            summary = (f"Сравнение {len(numeric_cols)} числовых колонок в таблице '{table_name}'. "
                       f"Наибольшее среднее значение в колонке '{max_mean_col}', "
                       f"наименьшее в '{min_mean_col}'.")
        else:
            summary = f"В таблице '{table_name}' нет числовых колонок для сравнения"

        return {
            'question': question,
            'data': clean_dataframe_for_json(comparison_df),
            'summary': summary,
            'analyzed_tables': [table_name],
            'chart_data': self._prepare_chart_data_safe(comparison_df, 'bar', 'column', 'mean')
        }

    def _prepare_chart_data_safe(self, df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> Optional[
        Dict[str, Any]]:
        """Безопасная подготовка данных для графика"""
        if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
            return None

        try:
            x_data = df[x_col].head(20).astype(str).tolist()
            y_data = []

            for val in df[y_col].head(20):
                if pd.isna(val):
                    y_data.append(0)
                else:
                    y_data.append(float(val))

            return {
                'type': chart_type,
                'x': x_data,
                'y': y_data,
                'title': f'{y_col} по {x_col}'
            }
        except Exception as e:
            logger.error(f"Ошибка подготовки данных для графика: {e}")
            return None

    def _prepare_chart_data_safe_from_list(self, data: List[Dict], chart_type: str, x_col: str, y_col: str) -> Optional[
        Dict[str, Any]]:
        """Безопасная подготовка данных для графика из списка"""
        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        try:
            x_data = []
            y_data = []

            for item in data[:20]:  # Ограничиваем количество
                if isinstance(item, dict) and x_col in item and y_col in item:
                    x_data.append(str(item[x_col]))

                    y_val = item[y_col]
                    if pd.isna(y_val):
                        y_data.append(0)
                    else:
                        y_data.append(float(y_val))

            if x_data and y_data:
                return {
                    'type': chart_type,
                    'x': x_data,
                    'y': y_data,
                    'title': f'{y_col} по {x_col}'
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Ошибка подготовки данных для графика из списка: {e}")
            return None

    def _create_table_chart(self, df: pd.DataFrame, table_name: str) -> Optional[Dict[str, Any]]:
        """Создает график для анализа таблицы"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
                # Гистограмма для первой числовой колонки
                col = numeric_cols[0]
                data_values = df[col].dropna().head(100).tolist()

                return {
                    'type': 'histogram',
                    'data': [float(x) for x in data_values],
                    'title': f'Распределение {col} в {table_name}'
                }
            else:
                # Круговая диаграмма для первой категориальной колонки
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    col = categorical_cols[0]
                    value_counts = df[col].value_counts().head(10)

                    return {
                        'type': 'pie',
                        'labels': [str(x) for x in value_counts.index.tolist()],
                        'values': [int(x) for x in value_counts.values.tolist()],
                        'title': f'Распределение {col} в {table_name}'
                    }

            return None

        except Exception as e:
            logger.error(f"Ошибка создания графика для таблицы: {e}")
            return None

    def _create_aggregation_chart(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Создает график для агрегированных данных"""
        if df.empty:
            return None

        try:
            if 'table' in df.columns:
                if 'count' in df.columns:
                    # Группируем по таблицам
                    table_counts = df.groupby('table')['count'].sum().reset_index()

                    return {
                        'type': 'bar',
                        'x': [str(x) for x in table_counts['table'].tolist()],
                        'y': [int(x) for x in table_counts['count'].tolist()],
                        'title': 'Агрегированные данные по таблицам'
                    }
                elif 'sum' in df.columns:
                    # Используем sum
                    table_sums = df.groupby('table')['sum'].sum().reset_index()

                    return {
                        'type': 'bar',
                        'x': [str(x) for x in table_sums['table'].tolist()],
                        'y': [float(x) for x in table_sums['sum'].tolist()],
                        'title': 'Суммы по таблицам'
                    }

            return None

        except Exception as e:
            logger.error(f"Ошибка создания графика агрегации: {e}")
            return None
