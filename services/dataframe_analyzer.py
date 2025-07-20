# services/dataframe_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from services.dataframe_manager import DataFrameManager
from datetime import datetime
import re

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
            else:
                return self._analyze_general(question)

        except Exception as e:
            logger.error(f"Ошибка анализа вопроса '{question}': {e}")
            return {
                'question': question,
                'error': str(e),
                'data': pd.DataFrame(),
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
            null_count = df.isnull().sum().sum()

            overview_data.append({
                'table_name': table_name,
                'row_count': len(df),
                'column_count': len(df.columns),
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'null_values': int(null_count),
                'null_percentage': round((null_count / (len(df) * len(df.columns))) * 100, 2),
                'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            })

        overview_df = pd.DataFrame(overview_data)

        # Информация о связях
        relations_info = []
        for rel in self.df_manager.relations:
            relations_info.append(f"{rel.from_table}.{rel.from_column} -> {rel.to_table}.{rel.to_column}")

        total_rows = overview_df['row_count'].sum()
        total_tables = len(self.df_manager.tables)
        relations_count = len(self.df_manager.relations)

        summary = (f"База данных содержит {total_tables} таблиц с {total_rows} записями. "
                   f"Обнаружено {relations_count} связей между таблицами. "
                   f"Общий объем данных в памяти: {self.df_manager.total_memory_usage:.2f} MB.")

        return {
            'question': 'Общий обзор базы данных',
            'data': overview_df,
            'summary': summary,
            'analyzed_tables': list(self.df_manager.tables.keys()),
            'relations': relations_info,
            'chart_data': self._prepare_chart_data(overview_df, 'bar', 'table_name', 'row_count')
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
                    'data': pd.DataFrame(),
                    'summary': f'Таблица {table_name} не найдена в загруженных данных',
                    'analyzed_tables': []
                }

        df = self.df_manager.tables[table_name]

        # Базовая информация
        basic_info = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }

        # Статистический анализ числовых колонок
        numeric_stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe().to_dict()

        # Анализ категориальных колонок
        categorical_stats = self._analyze_categorical_columns(df)

        # Поиск аномалий
        anomalies = self._detect_anomalies_dataframe(df)

        # Анализ пропущенных значений
        missing_analysis = self._analyze_missing_values(df)

        # Корреляционный анализ
        correlations = self._analyze_correlations_single_table(df)

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
            'data': df.head(20),
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
                'data': pd.DataFrame(),
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
                    'common_values_count': len(common_values),
                    'left_unique_count': len(left_unique),
                    'right_unique_count': len(right_unique),
                    'relationship_strength': round(relationship_strength, 2),
                    'integrity_issues': left_not_in_right,
                    'reverse_integrity_issues': right_not_in_left
                })

            except Exception as e:
                logger.error(f"Ошибка анализа связи {relation.from_table}->{relation.to_table}: {e}")

        if not relationship_data:
            return {
                'question': 'Анализ связей между таблицами',
                'data': pd.DataFrame(),
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
            'data': relationships_df,
            'summary': summary,
            'analyzed_tables': analyzed_tables,
            'chart_data': self._prepare_chart_data(relationships_df, 'bar',
                                                   'from_table',
                                                   'relationship_strength') if not relationships_df.empty else None
        }

    def _analyze_aggregations(self, question: str) -> Dict[str, Any]:
        """Анализ с агрегациями"""
        tables_mentioned = self._extract_mentioned_tables(question)

        if not tables_mentioned:
            # Берем самую большую таблицу
            tables_mentioned = [max(self.df_manager.tables.items(), key=lambda x: len(x[1]))[0]]

        results = []
        aggregated_data = []

        for table_name in tables_mentioned:
            df = self.df_manager.tables[table_name]

            # Числовые агрегации
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                agg_result = df[numeric_cols].agg(['count', 'sum', 'mean', 'min', 'max', 'std']).round(2)

                # Преобразуем в удобный формат
                for col in numeric_cols:
                    aggregated_data.append({
                        'table': table_name,
                        'column': col,
                        'type': 'numeric',
                        'count': agg_result.loc['count', col],
                        'sum': agg_result.loc['sum', col],
                        'mean': agg_result.loc['mean', col],
                        'min': agg_result.loc['min', col],
                        'max': agg_result.loc['max', col],
                        'std': agg_result.loc['std', col]
                    })

                results.append({
                    'table': table_name,
                    'type': 'numeric_aggregation',
                    'data': agg_result
                })

            # Категориальные агрегации
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols[:3]:  # Ограничиваем количество
                    value_counts = df[col].value_counts().head(10)
                    if len(value_counts) > 0:
                        for value, count in value_counts.items():
                            aggregated_data.append({
                                'table': table_name,
                                'column': col,
                                'type': 'categorical',
                                'value': str(value),
                                'count': count,
                                'percentage': round(count / len(df) * 100, 2)
                            })

                        results.append({
                            'table': table_name,
                            'type': 'categorical_aggregation',
                            'column': col,
                            'data': value_counts
                        })

        # Создаем итоговый DataFrame
        if aggregated_data:
            main_result = pd.DataFrame(aggregated_data)
            summary = f"Агрегированный анализ {len(tables_mentioned)} таблиц. Проанализировано {len(results)} различных метрик."
        else:
            main_result = pd.DataFrame()
            summary = "Не удалось выполнить агрегацию данных"

        return {
            'question': question,
            'data': main_result,
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'detailed_results': results,
            'chart_data': self._create_aggregation_chart(main_result) if not main_result.empty else None
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
                            first_val = trend_data['count'].iloc[0]
                            last_val = trend_data['count'].iloc[-1]
                            trend_direction = "рост" if last_val > first_val else "падение"
                            trend_percent = abs((last_val - first_val) / first_val * 100) if first_val > 0 else 0
                        else:
                            trend_direction = "недостаточно данных"
                            trend_percent = 0

                        trend_results.append({
                            'table': table_name,
                            'date_column': date_col,
                            'trend_data': trend_data,
                            'total_records': len(df_clean),
                            'periods_count': len(trend_data),
                            'trend_direction': trend_direction,
                            'trend_percent': round(trend_percent, 1),
                            'date_range': f"{df_clean[date_col].min()} - {df_clean[date_col].max()}"
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
            main_data = pd.DataFrame()
            summary = "Временные данные для анализа трендов не найдены"

        analyzed_tables = [r['table'] for r in trend_results]

        return {
            'question': question,
            'data': main_data,
            'summary': summary,
            'analyzed_tables': analyzed_tables,
            'all_trends': trend_results,
            'chart_data': self._prepare_chart_data(main_data, 'line', 'period_str',
                                                   'count') if not main_data.empty else None
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

            # Находим самые сильные корреляции
            strong_correlations = corr_df[corr_df['abs_correlation'] > 0.7]

            summary = (f"Анализ корреляций в {len(tables_mentioned)} таблицах. "
                       f"Найдено {len(all_correlations)} корреляций, "
                       f"из них {len(strong_correlations)} сильных (>0.7).")

        else:
            corr_df = pd.DataFrame()
            summary = "Значимые корреляции не найдены"

        return {
            'question': question,
            'data': corr_df,
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'chart_data': self._prepare_chart_data(corr_df, 'scatter', 'column1',
                                                   'correlation') if not corr_df.empty else None
        }

    def _analyze_comparison(self, question: str) -> Dict[str, Any]:
        """Анализ сравнения между таблицами или колонками"""
        tables_mentioned = self._extract_mentioned_tables(question)

        if len(tables_mentioned) < 2:
            # Если упомянута одна таблица, сравниваем колонки внутри неё
            if len(tables_mentioned) == 1:
                return self._compare_columns_in_table(tables_mentioned[0], question)
            else:
                # Берем две самые большие таблицы
                largest_tables = sorted(self.df_manager.tables.items(),
                                        key=lambda x: len(x[1]), reverse=True)[:2]
                tables_mentioned = [t[0] for t in largest_tables]

        comparison_data = []

        for table_name in tables_mentioned:
            df = self.df_manager.tables[table_name]

            # Базовая статистика таблицы
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            stats = {
                'table': table_name,
                'rows': len(df),
                'columns': len(df.columns),
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols),
                'null_values': int(df.isnull().sum().sum()),
                'memory_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }

            # Добавляем статистику по числовым колонкам
            if len(numeric_cols) > 0:
                numeric_stats = df[numeric_cols].describe().mean()
                stats.update({
                    'avg_mean': round(numeric_stats['mean'], 2),
                    'avg_std': round(numeric_stats['std'], 2),
                    'avg_min': round(numeric_stats['min'], 2),
                    'avg_max': round(numeric_stats['max'], 2)
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
            'data': comparison_df,
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'chart_data': self._prepare_chart_data(comparison_df, 'bar', 'table',
                                                   'rows') if not comparison_df.empty else None
        }

    def _analyze_general(self, question: str) -> Dict[str, Any]:
        """Общий анализ для неопределенных вопросов"""
        # Пытаемся извлечь упомянутые таблицы
        tables_mentioned = self._extract_mentioned_tables(question)

        if not tables_mentioned:
            # Берем самую большую таблицу
            largest_table = max(self.df_manager.tables.items(), key=lambda x: len(x[1]))
            tables_mentioned = [largest_table[0]]

        # Выполняем базовый анализ первой упомянутой таблицы
        table_name = tables_mentioned[0]
        df = self.df_manager.tables[table_name]

        # Создаем общую статистику
        general_stats = {
            'table_name': table_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns_list': list(df.columns),
            'data_types': df.dtypes.value_counts().to_dict(),
            'missing_values': int(df.isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum())
        }

        # Добавляем примеры данных
        sample_data = df.head(10)

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
        return max(self.df_manager.tables.items(), key=lambda x: len(x[1]))[0]

    def _extract_mentioned_tables(self, question: str) -> List[str]:
        """Извлекает все упомянутые таблицы"""
        mentioned = []
        question_lower = question.lower()

        for table_name in self.df_manager.tables.keys():
            if table_name.lower() in question_lower:
                mentioned.append(table_name)

        # Если ничего не найдено, возвращаем самую большую таблицу
        if not mentioned:
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
                        pd.to_datetime(sample, errors='raise')
                        date_cols.append(col)
                except:
                    continue

        return date_cols

    def _analyze_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ категориальных колонок"""
        categorical_analysis = {}
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            value_counts = df[col].value_counts()
            categorical_analysis[col] = {
                'unique_count': len(value_counts),
                'top_values': value_counts.head(5).to_dict(),
                'null_count': int(df[col].isnull().sum()),
                'most_common': str(value_counts.index[0]) if len(value_counts) > 0 else 'N/A',
                'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            }

        return categorical_analysis

    def _detect_anomalies_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение аномалий в DataFrame"""
        anomalies = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].count() > 10:  # Минимум данных для анализа
                try:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

                    if len(outliers) > 0:
                        anomalies.append({
                            'column': col,
                            'count': len(outliers),
                            'percentage': round(len(outliers) / len(df) * 100, 2),
                            'lower_bound': round(lower_bound, 2),
                            'upper_bound': round(upper_bound, 2),
                            'sample_values': outliers[col].head(5).tolist()
                        })
                except Exception as e:
                    logger.error(f"Ошибка обнаружения аномалий в колонке {col}: {e}")

        return anomalies

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ пропущенных значений"""
        missing_data = df.isnull().sum()
        total_missing = missing_data.sum()

        missing_analysis = {
            'total_missing': int(total_missing),
            'missing_percentage': round(total_missing / (len(df) * len(df.columns)) * 100, 2),
            'columns_with_missing': {}
        }

        for col, missing_count in missing_data.items():
            if missing_count > 0:
                missing_analysis['columns_with_missing'][col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_count / len(df) * 100, 2)
                }

        return missing_analysis

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
                        if not np.isnan(corr_value) and abs(corr_value) > 0.5:  # Значимая корреляция
                            correlations.append({
                                'table': table_name or 'unknown',
                                'column1': corr_matrix.columns[i],
                                'column2': corr_matrix.columns[j],
                                'correlation': round(corr_value, 3),
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
            col_stats = {
                'column': col,
                'count': int(df[col].count()),
                'mean': round(df[col].mean(), 2),
                'median': round(df[col].median(), 2),
                'std': round(df[col].std(), 2),
                'min': round(df[col].min(), 2),
                'max': round(df[col].max(), 2),
                'range': round(df[col].max() - df[col].min(), 2)
            }
            comparison_data.append(col_stats)

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
            'data': comparison_df,
            'summary': summary,
            'analyzed_tables': [table_name],
            'chart_data': self._prepare_chart_data(comparison_df, 'bar', 'column',
                                                   'mean') if not comparison_df.empty else None
        }

    def _prepare_chart_data(self, df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> Dict[str, Any]:
        """Подготавливает данные для создания графика"""
        if df.empty or x_col not in df.columns or y_col not in df.columns:
            return None

        try:
            return {
                'type': chart_type,
                'x': df[x_col].head(20).astype(str).tolist(),
                'y': df[y_col].head(20).tolist(),
                'title': f'{y_col} по {x_col}'
            }
        except Exception as e:
            logger.error(f"Ошибка подготовки данных для графика: {e}")
            return None

    def _create_table_chart(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Создает график для анализа таблицы"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            # Гистограмма для первой числовой колонки
            col = numeric_cols[0]
            return {
                'type': 'histogram',
                'data': df[col].dropna().tolist()[:100],  # Ограничиваем количество
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
                    'labels': value_counts.index.tolist(),
                    'values': value_counts.values.tolist(),
                    'title': f'Распределение {col} в {table_name}'
                }

        return None

    def _create_aggregation_chart(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Создает график для агрегированных данных"""
        if 'count' in df.columns and 'table' in df.columns:
            return {
                'type': 'bar',
                'x': df['table'].unique().tolist(),
                'y': [df[df['table'] == table]['count'].sum() for table in df['table'].unique()],
                'title': 'Агрегированные данные по таблицам'
            }

        return None
