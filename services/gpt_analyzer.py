# services/smart_gpt_analyzer.py - обновлен под существующую архитектуру

import openai
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import json
import config
import re

logger = logging.getLogger(__name__)


class SmartGPTAnalyzer:
    """Умный GPT-аналитик для интеграции с существующей системой"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=config.API_KEY)
        self.model = "gpt-4o-mini"

    def analyze_data_with_gpt(self, df: pd.DataFrame, table_name: str,
                              analysis_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Основной метод для GPT анализа (совместим с существующим интерфейсом)"""

        if context is None:
            context = {}

        # Создаем умный анализ на основе типа
        if analysis_type == "business_insights":
            return self._analyze_business_insights(df, table_name, context)
        elif analysis_type == "data_quality":
            return self._analyze_data_quality(df, table_name, context)
        elif analysis_type == "statistical_insights":
            return self._analyze_statistical_patterns(df, table_name, context)
        elif analysis_type == "predictive_analysis":
            return self._analyze_predictive_potential(df, table_name, context)
        else:
            return self._analyze_general_insights(df, table_name, context)

    def _analyze_business_insights(self, df: pd.DataFrame, table_name: str,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Бизнес-инсайты через GPT"""

        # Подготавливаем данные для анализа
        data_summary = self._prepare_data_summary(df, table_name)
        business_context = self._detect_business_domain(df, table_name)

        prompt = f"""
Проанализируй данные таблицы '{table_name}' и дай практические бизнес-инсайты.

КОНТЕКСТ ДАННЫХ:
{data_summary}

БИЗНЕС-ДОМЕН: {business_context}

ЗАДАЧА:
1. Найди 3-4 ключевых бизнес-паттерна
2. Предложи конкретные действия для улучшения
3. Определи потенциальные риски
4. Найди возможности для роста

Отвечай структурированно и конкретно. Фокусируйся на actionable инсайтах.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "Ты - опытный бизнес-аналитик. Даешь конкретные, actionable рекомендации на основе данных."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )

            analysis = response.choices[0].message.content

            return {
                "gpt_analysis": analysis,
                "confidence": "high",
                "analysis_type": "business_insights",
                "business_context": business_context
            }

        except Exception as e:
            logger.error(f"Ошибка GPT бизнес-анализа: {e}")
            return {
                "gpt_analysis": f"Бизнес-анализ недоступен: {str(e)}",
                "confidence": "low",
                "analysis_type": "business_insights"
            }

    def _analyze_data_quality(self, df: pd.DataFrame, table_name: str,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ качества данных через GPT"""

        # Вычисляем метрики качества
        quality_metrics = self._calculate_quality_metrics(df)

        prompt = f"""
Оцени качество данных в таблице '{table_name}' и дай рекомендации по улучшению.

МЕТРИКИ КАЧЕСТВА:
- Записей: {len(df):,}
- Колонок: {len(df.columns)}
- Пропущенных значений: {quality_metrics['missing_percentage']:.1f}%
- Дубликатов: {quality_metrics['duplicates_percentage']:.1f}%
- Числовых колонок: {quality_metrics['numeric_columns']}
- Категориальных: {quality_metrics['categorical_columns']}

ПРОБЛЕМЫ:
{json.dumps(quality_metrics['issues'], indent=2, ensure_ascii=False)}

ЗАДАЧА:
1. Оцени общее качество данных (1-10)
2. Определи критичные проблемы
3. Дай приоритетные рекомендации по улучшению
4. Предложи план действий

Будь конкретным и практичным.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "Ты - эксперт по качеству данных. Даешь практические рекомендации по улучшению качества."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )

            return {
                "gpt_analysis": response.choices[0].message.content,
                "confidence": "high",
                "analysis_type": "data_quality",
                "quality_score": quality_metrics['overall_score']
            }

        except Exception as e:
            logger.error(f"Ошибка GPT анализа качества: {e}")
            return {
                "gpt_analysis": f"Анализ качества данных недоступен: {str(e)}",
                "confidence": "low",
                "analysis_type": "data_quality"
            }

    def _prepare_data_summary(self, df: pd.DataFrame, table_name: str) -> str:
        """Подготавливает краткую сводку данных"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns

            summary_parts = [
                f"Таблица: {table_name}",
                f"Записей: {len(df):,}",
                f"Колонок: {len(df.columns)} ({len(numeric_cols)} числовых, {len(categorical_cols)} категориальных)"
            ]

            # Добавляем примеры колонок
            if len(df.columns) > 0:
                summary_parts.append(f"Колонки: {', '.join(df.columns[:5].tolist())}")
                if len(df.columns) > 5:
                    summary_parts.append("...")

            # Добавляем основные статистики для числовых колонок
            if len(numeric_cols) > 0:
                numeric_summary = df[numeric_cols].describe()
                key_stats = []
                for col in numeric_cols[:3]:  # Первые 3 числовые колонки
                    mean_val = numeric_summary.loc['mean', col]
                    if not pd.isna(mean_val):
                        key_stats.append(f"{col}: среднее {mean_val:.2f}")

                if key_stats:
                    summary_parts.append("Ключевые показатели: " + ", ".join(key_stats))

            return "\n".join(summary_parts)

        except Exception as e:
            logger.error(f"Ошибка создания сводки данных: {e}")
            return f"Таблица {table_name}: {len(df)} записей, {len(df.columns)} колонок"

    def _detect_business_domain(self, df: pd.DataFrame, table_name: str) -> str:
        """Автоматически определяет бизнес-домен"""

        # Анализируем название таблицы и колонки
        all_text = (table_name + " " + " ".join(df.columns)).lower()

        if any(word in all_text for word in ['user', 'customer', 'client', 'visitor', 'person']):
            return "customer_analytics"
        elif any(word in all_text for word in ['sale', 'order', 'purchase', 'transaction', 'payment']):
            return "sales_analytics"
        elif any(word in all_text for word in ['product', 'item', 'inventory', 'stock']):
            return "product_analytics"
        elif any(word in all_text for word in ['employee', 'staff', 'hr', 'salary', 'department']):
            return "hr_analytics"
        elif any(word in all_text for word in ['revenue', 'profit', 'cost', 'price', 'financial']):
            return "financial_analytics"
        elif any(word in all_text for word in ['marketing', 'campaign', 'ad', 'conversion']):
            return "marketing_analytics"
        else:
            return "general_business"

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Вычисляет метрики качества данных"""

        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()

        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object']).columns)

        # Ищем проблемы
        issues = []

        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        if missing_pct > 10:
            issues.append(f"Высокий процент пропусков: {missing_pct:.1f}%")

        duplicate_pct = (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0
        if duplicate_pct > 5:
            issues.append(f"Много дубликатов: {duplicate_pct:.1f}%")

        # Проверяем числовые колонки на аномалии
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].min() < 0 and 'id' not in col.lower() and 'price' in col.lower():
                issues.append(f"Отрицательные цены в колонке {col}")

        # Общая оценка качества
        overall_score = 100
        overall_score -= missing_pct * 0.5  # штраф за пропуски
        overall_score -= duplicate_pct * 0.3  # штраф за дубликаты
        overall_score -= len(issues) * 5  # штраф за каждую проблему
        overall_score = max(0, min(100, overall_score))

        return {
            "missing_percentage": missing_pct,
            "duplicates_percentage": duplicate_pct,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "issues": issues,
            "overall_score": round(overall_score, 1)
        }

    def generate_executive_summary(self, findings: List[Dict[str, Any]],
                                   table_summary: Dict[str, Any]) -> str:
        """Генерирует executive summary (совместимо с существующим интерфейсом)"""

        # Собираем ключевые инсайты
        key_insights = []
        for finding in findings:
            gpt_data = finding.get('gpt_insights', {})
            if gpt_data and gpt_data.get('gpt_analysis'):
                # Берем первые 200 символов каждого инсайта
                insight = gpt_data['gpt_analysis'][:200] + "..." if len(gpt_data['gpt_analysis']) > 200 else gpt_data[
                    'gpt_analysis']
                key_insights.append(insight)

        if not key_insights:
            return f"Завершен анализ {table_summary.get('total_tables', 0)} таблиц. Получены детальные технические показатели и паттерны данных."

        prompt = f"""
Создай executive summary для руководства на основе проведенного анализа данных:

КЛЮЧЕВЫЕ НАХОДКИ:
{chr(10).join(key_insights[:3])}

СТАТИСТИКА АНАЛИЗА:
- Таблиц проанализировано: {table_summary.get('total_tables', 0)}
- Связей найдено: {table_summary.get('total_relations', 0)}
- Объем данных: {table_summary.get('total_memory_mb', 0):.1f} MB

ЗАДАЧА:
Создай краткое (2-3 абзаца) executive summary для руководителя.
Фокус на:
1. Главные бизнес-выводы
2. Критичные действия
3. Потенциальные возможности

Пиши по-русски, кратко и по делу.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "Ты создаешь executive summary для руководства. Пиши четко, кратко, акцент на ROI."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Ошибка создания executive summary: {e}")
            return f"Проведен комплексный анализ {table_summary.get('total_tables', 0)} таблиц. Выявлены ключевые паттерны и возможности для оптимизации бизнес-процессов. Детальные рекомендации доступны в разделах анализа."

    def analyze_correlations_with_context(self, correlations: List[Dict],
                                          df: pd.DataFrame, table_name: str) -> str:
        """Анализирует корреляции с бизнес-контекстом (совместимо с существующим API)"""

        if not correlations:
            return "Значимые корреляции не обнаружены"

        prompt = f"""
Проанализируй найденные корреляции в таблице '{table_name}' и дай бизнес-интерпретацию:

НАЙДЕННЫЕ КОРРЕЛЯЦИИ:
{json.dumps(correlations, indent=2, ensure_ascii=False)}

ЗАДАЧА:
1. Объясни практическое значение каждой корреляции
2. Какие бизнес-решения можно принять на основе этих связей?
3. На что обратить внимание руководству?

Отвечай кратко и по делу.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "Ты интерпретируешь статистические корреляции для бизнеса. Объясняешь простым языком."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Ошибка анализа корреляций: {e}")
            return f"Найдено {len(correlations)} корреляций. GPT-интерпретация недоступна: {str(e)}"

    # Дополнительные методы для совместимости
    def _analyze_statistical_patterns(self, df: pd.DataFrame, table_name: str,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ статистических паттернов"""

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return {
                "gpt_analysis": "В данных отсутствуют числовые колонки для статистического анализа",
                "confidence": "low",
                "analysis_type": "statistical_insights"
            }

        # Вычисляем основные статистики
        stats_summary = df[numeric_cols].describe()

        prompt = f"""
Проанализируй статистические паттерны в таблице '{table_name}':

СТАТИСТИКИ:
{stats_summary.to_string()}

ЗАДАЧА:
1. Найди необычные паттерны в распределениях
2. Определи выбросы и аномалии
3. Дай рекомендации по дальнейшему анализу
4. Предложи метрики для мониторинга

Фокусируйся на практических выводах.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты - статистик-аналитик. Интерпретируешь статистики для бизнеса."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )

            return {
                "gpt_analysis": response.choices[0].message.content,
                "confidence": "high",
                "analysis_type": "statistical_insights"
            }

        except Exception as e:
            logger.error(f"Ошибка статистического анализа: {e}")
            return {
                "gpt_analysis": f"Статистический анализ недоступен: {str(e)}",
                "confidence": "low",
                "analysis_type": "statistical_insights"
            }

    def _analyze_predictive_potential(self, df: pd.DataFrame, table_name: str,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ предиктивного потенциала"""

        # Ищем временные колонки
        date_cols = [col for col in df.columns
                     if any(date_word in col.lower()
                            for date_word in ['date', 'time', 'created', 'updated'])]

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        prompt = f"""
Оцени предиктивный потенциал данных в таблице '{table_name}':

СТРУКТУРА ДАННЫХ:
- Записей: {len(df):,}
- Числовых колонок: {len(numeric_cols)}
- Временных колонок: {len(date_cols)}
- Колонки: {', '.join(df.columns[:10].tolist())}

ЗАДАЧА:
1. Можно ли построить прогнозные модели?
2. Какие метрики лучше всего подходят для прогнозирования?
3. Какие дополнительные данные нужны?
4. Практические применения предиктивной аналитики

Дай конкретные рекомендации.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "Ты - ML-инженер. Оцениваешь данные для машинного обучения и прогнозирования."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            return {
                "gpt_analysis": response.choices[0].message.content,
                "confidence": "medium",
                "analysis_type": "predictive_analysis",
                "predictive_score": self._calculate_predictive_score(df, date_cols, numeric_cols)
            }

        except Exception as e:
            logger.error(f"Ошибка предиктивного анализа: {e}")
            return {
                "gpt_analysis": f"Предиктивный анализ недоступен: {str(e)}",
                "confidence": "low",
                "analysis_type": "predictive_analysis"
            }

    def _calculate_predictive_score(self, df: pd.DataFrame, date_cols: List[str],
                                    numeric_cols: List[str]) -> int:
        """Вычисляет потенциал для предиктивного моделирования (0-100)"""

        score = 0

        # Базовые баллы за структуру данных
        if len(df) > 1000:
            score += 30
        elif len(df) > 100:
            score += 15

        # Баллы за временные данные
        if date_cols:
            score += 25

        # Баллы за числовые данные
        if len(numeric_cols) > 5:
            score += 20
        elif len(numeric_cols) > 2:
            score += 10

        # Баллы за качество данных
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct < 10:
            score += 15
        elif missing_pct < 25:
            score += 5

        # Баллы за разнообразие данных
        if len(df.columns) > 10:
            score += 10

        return min(100, score)

    def _analyze_general_insights(self, df: pd.DataFrame, table_name: str,
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Общий анализ данных"""

        data_summary = self._prepare_data_summary(df, table_name)

        prompt = f"""
Проанализируй данные и дай общие бизнес-инсайты:

{data_summary}

ЗАДАЧА:
1. Что можно сказать о качестве данных?
2. Какие паттерны видны в структуре?
3. Есть ли явные проблемы или возможности?
4. Следующие шаги для анализа?

Будь конкретным и практичным.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты - бизнес-аналитик. Даешь общие инсайты по данным."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=700
            )

            return {
                "gpt_analysis": response.choices[0].message.content,
                "confidence": "medium",
                "analysis_type": "general_insights"
            }

        except Exception as e:
            logger.error(f"Ошибка общего анализа: {e}")
            return {
                "gpt_analysis": f"Общий анализ недоступен: {str(e)}",
                "confidence": "low",
                "analysis_type": "general_insights"
            }
