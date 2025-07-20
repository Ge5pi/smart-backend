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
from services.gpt_analyzer import SmartGPTAnalyzer  # Обновленный импорт

# Добавляем путь для импорта utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_serializer import convert_to_serializable, clean_dataframe_for_json

logger = logging.getLogger(__name__)


class DataFrameAnalyzer:
    """Аналитический движок для работы с DataFrame"""

    def __init__(self, df_manager: DataFrameManager):
        self.df_manager = df_manager
        self.analysis_cache = {}
        self.smart_gpt = SmartGPTAnalyzer()

    def _detect_user_focus(self, question: str) -> str:
        """Определяет основной фокус пользователя"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['проблем', 'ошибк', 'аномали', 'неправиль']):
            return 'problem_solving'
        elif any(word in question_lower for word in ['возможност', 'потенциал', 'рост', 'улучшен']):
            return 'opportunity_discovery'
        elif any(word in question_lower for word in ['сравн', 'различ', 'vs', 'против']):
            return 'comparative_analysis'
        elif any(word in question_lower for word in ['тренд', 'динамик', 'изменен', 'развити']):
            return 'trend_analysis'
        else:
            return 'general_insights'

    def _analyze_statistical_insights(self, question: str) -> Dict[str, Any]:
        """Углубленный статистический анализ с GPT интерпретацией"""
        tables_mentioned = self._extract_mentioned_tables(question)
        if not tables_mentioned:
            tables_mentioned = [max(self.df_manager.tables.items(), key=lambda x: len(x[1]))[0]]

        main_table = tables_mentioned[0]
        df = self.df_manager.tables[main_table]

        # Расширенный статистический анализ
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        statistical_results = []

        if len(numeric_cols) > 0:
            # Основная статистика
            desc_stats = df[numeric_cols].describe()

            # Дополнительные метрики
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 5:
                    skewness = col_data.skew()
                    kurtosis = col_data.kurtosis()

                    statistical_results.append({
                        'column': col,
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'skewness': float(skewness),
                        'kurtosis': float(kurtosis),
                        'distribution_type': self._classify_distribution(skewness, kurtosis),
                        'outliers_count': len(self._detect_outliers_iqr(col_data)),
                        'coefficient_variation': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0
                    })

        stats_df = pd.DataFrame(statistical_results)

        # Корреляционная матрица с интерпретацией
        correlations = self._analyze_correlations_single_table(df, main_table)

        summary = f"📊 Статистический анализ таблицы '{main_table}' завершен"

        return {
            'question': question,
            'data': clean_dataframe_for_json(stats_df),
            'summary': summary,
            'analyzed_tables': [main_table],
            'statistical_insights': statistical_results,
            'correlations': correlations,
            'chart_data': self._prepare_chart_data_safe(stats_df, 'bar', 'column', 'mean')
        }

    def _detect_outliers_iqr(self, data: pd.Series) -> List:
        """Обнаружение выбросов методом IQR"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data < lower_bound) | (data > upper_bound)].tolist()

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

    def _create_business_context(self, question: str, analysis_type: str) -> Dict[str, Any]:
        """Создает бизнес-контекст для умного GPT-анализа"""
        context = {
            'question_intent': question,
            'analysis_type': analysis_type,
            'user_focus': self._detect_user_focus(question)
        }

        # Добавляем информацию о домене на основе названий таблиц
        table_names = list(self.df_manager.tables.keys())
        context['entities'] = table_names

        # Определяем приоритетные метрики на основе вопроса
        question_lower = question.lower()
        if any(word in question_lower for word in ['прибыль', 'доход', 'стоимость', 'цена']):
            context['priority_metrics'] = ['financial_performance', 'profitability']
        elif any(word in question_lower for word in ['клиент', 'пользователь', 'покупатель']):
            context['priority_metrics'] = ['customer_satisfaction', 'retention']
        elif any(word in question_lower for word in ['продажи', 'конверсия', 'воронка']):
            context['priority_metrics'] = ['sales_performance', 'conversion_rates']
        else:
            context['priority_metrics'] = ['general_performance']

        return context

    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Анализирует вопрос и возвращает результат с умным GPT-анализом"""
        logger.info(f"Анализ вопроса: {question}")

        # Определяем тип анализа
        analysis_type = self._categorize_question(question)

        try:
            # Выполняем базовый DataFrame анализ
            if analysis_type == 'overview':
                base_results = self._analyze_overview()
            elif analysis_type == 'table_analysis':
                table_name = self._extract_table_name(question)
                base_results = self._analyze_single_table(table_name)
            elif analysis_type == 'relationship_analysis':
                base_results = self._analyze_relationships()
            elif analysis_type == 'aggregation':
                base_results = self._analyze_aggregations(question)
            elif analysis_type == 'trend_analysis':
                base_results = self._analyze_trends(question)
            elif analysis_type == 'correlation':
                base_results = self._analyze_correlations(question)
            elif analysis_type == 'comparison':
                base_results = self._analyze_comparison(question)
            elif analysis_type == 'anomalies':
                base_results = self._analyze_anomalies(question)
            elif analysis_type == 'business_insights':
                base_results = self._analyze_business_metrics(question)
            elif analysis_type == 'statistical_insights':
                base_results = self._analyze_statistical_insights(question)
            elif analysis_type == 'predictive_analysis':
                base_results = self._analyze_predictive_patterns(question)
            elif analysis_type == 'data_quality':
                base_results = self._analyze_data_quality_comprehensive(question)
            else:
                base_results = self._analyze_general(question)

            # Применяем умный GPT-анализ к результатам
            if not base_results.get('error') and base_results.get('data'):
                try:
                    # Создаем бизнес-контекст для GPT
                    business_context = self._create_business_context(question, analysis_type)

                    # Получаем умные инсайты
                    smart_analysis = self.smart_gpt.analyze_findings_with_context(
                        dataframe_results=base_results,
                        business_context=business_context
                    )

                    # Обогащаем результаты GPT-инсайтами
                    base_results['smart_gpt_insights'] = smart_analysis
                    base_results['business_insights'] = smart_analysis.get('business_insights', '')
                    base_results['action_items'] = smart_analysis.get('action_items', [])
                    base_results['risk_assessment'] = smart_analysis.get('risk_assessment', '')
                    base_results['opportunities'] = smart_analysis.get('opportunities', [])
                    base_results['gpt_confidence'] = smart_analysis.get('confidence', 'medium')

                    # Обновляем summary с бизнес-инсайтами
                    if smart_analysis.get('business_insights'):
                        base_results['summary'] = (
                            f"**БИЗНЕС-АНАЛИЗ:**\n{smart_analysis['business_insights']}\n\n"
                            f"**ТЕХНИЧЕСКИЕ ДАННЫЕ:**\n{base_results.get('summary', '')}"
                        )

                except Exception as gpt_error:
                    logger.error(f"Ошибка умного GPT-анализа: {gpt_error}")
                    base_results['smart_gpt_insights'] = {
                        'business_insights': 'GPT-анализ временно недоступен',
                        'confidence': 'low'
                    }

            return base_results

        except Exception as e:
            logger.error(f"Ошибка анализа вопроса '{question}': {e}")
            return {
                'question': question,
                'error': str(e),
                'data': [],
                'summary': f'Не удалось проанализировать: {str(e)}',
                'analyzed_tables': []
            }

    def _analyze_data_quality_comprehensive(self, question: str) -> Dict[str, Any]:
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

            # Анализ по типам колонок
            numeric_quality = self._analyze_numeric_quality(df)
            categorical_quality = self._analyze_categorical_quality(df)

            # Целостность данных
            integrity_issues = []

            # Проверка на отрицательные значения где они не должны быть
            for col in df.select_dtypes(include=[np.number]).columns:
                if 'price' in col.lower() or 'amount' in col.lower() or 'quantity' in col.lower():
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        integrity_issues.append(f"{col}: {negative_count} отрицательных значений")

            quality_score = self._calculate_quality_score(
                null_percentage=null_cells / total_cells * 100,
                duplicate_percentage=duplicate_rows / len(df) * 100,
                integrity_issues_count=len(integrity_issues)
            )

            quality_results.append({
                'table': table_name,
                'quality_score': round(quality_score, 1),
                'completeness': round((1 - null_cells / total_cells) * 100, 1),
                'uniqueness': round((1 - duplicate_rows / len(df)) * 100, 1),
                'integrity_issues': len(integrity_issues),
                'numeric_quality': numeric_quality,
                'categorical_quality': categorical_quality,
                'recommendations': self._generate_quality_recommendations(
                    null_cells / total_cells * 100, duplicate_rows / len(df) * 100, integrity_issues
                )
            })

        quality_df = pd.DataFrame(quality_results)

        avg_quality = quality_df['quality_score'].mean() if not quality_df.empty else 0
        summary = f"🔍 Анализ качества данных: средний балл качества {avg_quality:.1f}/100"

        return {
            'question': question,
            'data': clean_dataframe_for_json(quality_df),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'quality_metrics': quality_results,
            'chart_data': self._prepare_chart_data_safe(quality_df, 'bar', 'table', 'quality_score')
        }

    def _generate_quality_recommendations(self, null_pct: float, duplicate_pct: float, issues: List[str]) -> List[str]:
        """Генерирует рекомендации по улучшению качества данных"""
        recommendations = []

        if null_pct > 10:
            recommendations.append("Критически высокий уровень пропусков - требуется стратегия заполнения")
        elif null_pct > 5:
            recommendations.append("Умеренный уровень пропусков - рассмотрите импутацию")

        if duplicate_pct > 5:
            recommendations.append("Обнаружены дубликаты - требуется дедупликация")

        if issues:
            recommendations.append("Найдены проблемы целостности данных - требуется валидация")

        if not recommendations:
            recommendations.append("Качество данных в норме")

        return recommendations

    def _analyze_categorical_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ качества категориальных данных"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) == 0:
            return {'columns_analyzed': 0}

        categorical_issues = []
        for col in categorical_cols:
            # Проверка на слишком много уникальных значений
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.8:
                categorical_issues.append(f"{col}: подозрительно много уникальных значений ({unique_ratio:.1%})")

        return {
            'columns_analyzed': len(categorical_cols),
            'issues_found': len(categorical_issues),
            'issues': categorical_issues
        }

    def _calculate_quality_score(self, null_percentage: float, duplicate_percentage: float,
                                 integrity_issues_count: int) -> float:
        """Рассчитывает общий балл качества данных"""
        base_score = 100.0

        # Снижаем за пропуски
        base_score -= null_percentage * 0.5

        # Снижаем за дубликаты
        base_score -= duplicate_percentage * 0.3

        # Снижаем за проблемы целостности
        base_score -= integrity_issues_count * 5

        return max(0.0, base_score)


    def _analyze_numeric_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ качества числовых данных"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {'columns_analyzed': 0}

        numeric_issues = []
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                numeric_issues.append(f"{col}: {inf_count} бесконечных значений")

        return {
            'columns_analyzed': len(numeric_cols),
            'issues_found': len(numeric_issues),
            'issues': numeric_issues
        }

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

                    if len(df_clean) > 10:  # Минимум данных для анализа
                        # Анализ трендов по месяцам
                        df_clean['period'] = df_clean[date_col].dt.to_period('M')

                        for num_col in numeric_cols[:3]:  # Ограничиваем для производительности
                            monthly_data = df_clean.groupby('period')[num_col].agg(
                                ['count', 'sum', 'mean']).reset_index()

                            if len(monthly_data) > 3:
                                # Простой анализ тренда
                                trend_data = monthly_data['mean'].values
                                trend_direction = 'растущий' if trend_data[-1] > trend_data[0] else 'убывающий'

                                # Сезонность (упрощенная)
                                seasonality_score = np.std(trend_data) / np.mean(trend_data) if np.mean(
                                    trend_data) > 0 else 0

                                predictive_insights.append({
                                    'table': table_name,
                                    'metric': num_col,
                                    'trend_direction': trend_direction,
                                    'seasonality_score': round(float(seasonality_score), 3),
                                    'periods_analyzed': len(monthly_data),
                                    'predictability': 'высокая' if seasonality_score < 0.3 else 'средняя' if seasonality_score < 0.6 else 'низкая',
                                    'data_points': len(df_clean)
                                })

                except Exception as e:
                    logger.error(f"Ошибка анализа временного ряда в {table_name}: {e}")

        predictive_df = pd.DataFrame(predictive_insights)

        if not predictive_df.empty:
            summary = f"🔮 Анализ предсказательных паттернов: найдено {len(predictive_insights)} временных рядов"
        else:
            summary = "Временные данные для предсказательного анализа не найдены"

        return {
            'question': question,
            'data': clean_dataframe_for_json(predictive_df),
            'summary': summary,
            'analyzed_tables': tables_mentioned,
            'predictive_patterns': predictive_insights,
            'chart_data': self._prepare_chart_data_safe(predictive_df, 'scatter', 'seasonality_score', 'data_points')
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

        logger.info(f"Запуск GPT анализа для таблицы {table_name}")

        gpt_business_insights = self.gpt_analyzer.analyze_data_with_gpt(
            df=df,
            table_name=table_name,
            analysis_type="business_insights"
        )

        gpt_data_quality = self.gpt_analyzer.analyze_data_with_gpt(
            df=df,
            table_name=table_name,
            analysis_type="data_quality"
        )

        # GPT анализ корреляций
        correlation_insights = ""
        if correlations:
            correlation_insights = self.gpt_analyzer.analyze_correlations_with_context(
                correlations, df, table_name
            )

        # Создаем обогащенную сводку
        summary = f"🎯 **GPT-Анализ таблицы '{table_name}'**\n\n"
        summary += f"📊 Базовая информация: {len(df):,} записей, {len(df.columns)} колонок\n\n"

        if anomalies:
            summary += f"⚠️ Обнаружено {len(anomalies)} типов аномалий\n"
        if correlations:
            summary += f"🔗 Найдено {len(correlations)} значимых корреляций\n"

        summary += "\n📈 **Бизнес-инсайты:**\n" + gpt_business_insights.get('gpt_analysis', 'Недоступно')

        return {
            'question': f'Детальный анализ таблицы {table_name}',
            'data': clean_dataframe_for_json(df.head(10)),
            'summary': summary,
            'analyzed_tables': [table_name],
            'gpt_business_insights': gpt_business_insights.get('gpt_analysis', ''),
            'gpt_data_quality': gpt_data_quality.get('gpt_analysis', ''),
            'correlation_insights': correlation_insights,
            'basic_info': basic_info,
            'anomalies': anomalies,
            'correlations': correlations
        }

    def _analyze_business_metrics(self, question: str) -> Dict[str, Any]:
        """НОВЫЙ метод: Анализ бизнес-метрик с GPT"""

        tables_mentioned = self._extract_mentioned_tables(question)
        if not tables_mentioned:
            tables_mentioned = [max(self.df_manager.tables.items(), key=lambda x: len(x[1]))[0]]

        main_table = tables_mentioned[0]
        df = self.df_manager.tables[main_table]

        # Вычисляем основные метрики
        metrics = self._calculate_business_metrics(df)

        # GPT анализ метрик
        gpt_analysis = self.gpt_analyzer.analyze_data_with_gpt(
            df=df,
            table_name=main_table,
            analysis_type="business_insights",
            context={"metrics": metrics, "question": question}
        )

        # Создаем результат
        metrics_df = pd.DataFrame([metrics])

        summary = f"🚀 **Бизнес-анализ таблицы '{main_table}'**\n\n"
        summary += gpt_analysis.get('gpt_analysis', 'GPT анализ недоступен')

        return {
            'question': question,
            'data': clean_dataframe_for_json(metrics_df),
            'summary': summary,
            'analyzed_tables': [main_table],
            'gpt_insights': gpt_analysis.get('gpt_analysis', ''),
            'business_metrics': metrics
        }

    def _calculate_business_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Вычисляет ключевые бизнес-метрики"""

        metrics = {
            'total_records': len(df),
            'data_completeness': round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1),
            'unique_entities': {}
        }

        # Анализ числовых колонок
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if 'value' in col.lower() or 'amount' in col.lower() or 'price' in col.lower():
                    metrics[f'{col}_total'] = float(df[col].sum())
                    metrics[f'{col}_average'] = round(float(df[col].mean()), 2)
                    metrics[f'{col}_median'] = round(float(df[col].median()), 2)

        # Анализ категориальных данных
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() < 50:  # Ограничение для производительности
                metrics['unique_entities'][col] = int(df[col].nunique())

        return metrics

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
