# services/gpt_analyzer.py
import openai
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
import json
from config import API_KEY

logger = logging.getLogger(__name__)


class GPTAnalyzer:
    """GPT-powered аналитик для глубокого анализа данных"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=API_KEY)
        self.model = "gpt-4o-mini"

    def analyze_data_with_gpt(self,
                              df: pd.DataFrame,
                              table_name: str,
                              analysis_type: str,
                              context: Dict = None) -> Dict[str, Any]:
        """Анализирует данные с помощью GPT"""

        # Подготавливаем контекст данных
        data_summary = self._prepare_data_summary(df, table_name)

        # Выбираем промпт в зависимости от типа анализа
        prompt = self._get_analysis_prompt(analysis_type, data_summary, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )

            analysis_result = response.choices[0].message.content

            return {
                "gpt_analysis": analysis_result,
                "data_summary": data_summary,
                "analysis_type": analysis_type,
                "confidence": "high"
            }

        except Exception as e:
            logger.error(f"Ошибка GPT анализа: {e}")
            return {
                "gpt_analysis": f"Не удалось выполнить GPT анализ: {str(e)}",
                "confidence": "low"
            }

    def _prepare_data_summary(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Подготавливает краткое описание данных для GPT"""

        summary = {
            "table_name": table_name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_info": {}
        }

        # Анализ каждой колонки
        for col in df.columns:
            col_info = {
                "type": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_percentage": round((df[col].isnull().sum() / len(df)) * 100, 1)
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": round(float(df[col].mean()), 2) if not pd.isna(df[col].mean()) else None,
                    "std": round(float(df[col].std()), 2) if not pd.isna(df[col].std()) else None
                })

                # Поиск выбросов
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                col_info["outliers_count"] = len(outliers)

            elif df[col].dtype == 'object':
                value_counts = df[col].value_counts()
                col_info.update({
                    "unique_values": int(df[col].nunique()),
                    "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                    "top_3_values": value_counts.head(3).to_dict()
                })

            summary["column_info"][col] = col_info

        # Добавляем примеры данных
        sample_data = df.head(5).to_dict('records')
        summary["sample_data"] = sample_data

        return summary

    def _get_system_prompt(self) -> str:
        """Системный промпт для GPT-аналитика"""
        return """Ты - эксперт-аналитик данных с глубокими знаниями в статистике, машинном обучении и бизнес-аналитике.

Твоя задача - провести профессиональный анализ данных и предоставить:
1. Детальные инсайты о данных
2. Выявление скрытых паттернов
3. Бизнес-рекомендации
4. Обнаружение аномалий и их возможные причины
5. Прогнозы и тренды

Анализируй данные как настоящий data scientist, используя статистические методы и здравый смысл.
Отвечай на русском языке, структурированно и профессионально."""

    def _get_analysis_prompt(self, analysis_type: str, data_summary: Dict, context: Dict = None) -> str:
        """Генерирует промпт в зависимости от типа анализа"""

        base_prompt = f"""
Данные для анализа:
Таблица: {data_summary['table_name']}
Размер: {data_summary['rows']} строк, {data_summary['columns']} колонок

Информация о колонках:
{json.dumps(data_summary['column_info'], indent=2, ensure_ascii=False)}

Примеры данных:
{json.dumps(data_summary['sample_data'], indent=2, ensure_ascii=False)}
"""

        if analysis_type == "business_insights":
            return base_prompt + """
ЗАДАЧА: Проведи бизнес-анализ этих данных.

Проанализируй и ответь на:
1. Какие ключевые бизнес-метрики можно извлечь из этих данных?
2. Какие тренды и паттерны ты видишь?
3. Есть ли проблемные области, требующие внимания?
4. Какие возможности для роста/оптимизации ты видишь?
5. Конкретные рекомендации для принятия решений

Фокусируйся на практических инсайтах, а не на технических деталях."""

        elif analysis_type == "data_quality":
            return base_prompt + """
ЗАДАЧА: Оцени качество данных и найди проблемы.

Проанализируй:
1. Общее качество данных (полнота, консистентность, точность)
2. Проблемы с пропущенными значениями - их причины и влияние
3. Выбросы и аномалии - что они могут означать?
4. Потенциальные ошибки в данных
5. Рекомендации по улучшению качества данных

Будь конкретен в выявлении проблем и предложении решений."""

        elif analysis_type == "statistical_insights":
            return base_prompt + """
ЗАДАЧА: Проведи углубленный статистический анализ.

Проанализируй:
1. Распределения данных - что они говорят нам?
2. Корреляции и взаимосвязи между переменными
3. Статистические аномалии и их интерпретация
4. Значимые паттерны в данных
5. Статистические выводы и их практическое значение

Объясни сложные статистические концепты простым языком."""

        elif analysis_type == "predictive_analysis":
            return base_prompt + """
ЗАДАЧА: Проведи предиктивный анализ данных.

Проанализируй:
1. Временные тренды и сезонность (если применимо)
2. Факторы, влияющие на ключевые метрики
3. Возможные будущие сценарии развития
4. Риски и возможности
5. Рекомендации для прогнозирования

Сосредоточься на практических прогнозах, а не на технических деталях моделей."""

        else:
            return base_prompt + """
ЗАДАЧА: Проведи комплексный анализ данных.

Дай профессиональную оценку данных, включая:
1. Ключевые находки и инсайты
2. Важные паттерны и аномалии
3. Бизнес-значение данных
4. Рекомендации по использованию
5. Области для дальнейшего исследования"""

    def analyze_correlations_with_context(self,
                                          correlations: List[Dict],
                                          df: pd.DataFrame,
                                          table_name: str) -> str:
        """Анализирует корреляции с помощью GPT для получения инсайтов"""

        if not correlations:
            return "Значимые корреляции не обнаружены."

        prompt = f"""
Проанализируй найденные корреляции в таблице '{table_name}':

Корреляции:
{json.dumps(correlations, indent=2, ensure_ascii=False)}

Размер данных: {len(df)} записей

ЗАДАЧА: Объясни что означают эти корреляции:
1. Какие из них имеют практическое значение?
2. Могут ли они указывать на причинно-следственные связи?
3. Есть ли неожиданные или интересные корреляции?
4. Как эти корреляции можно использовать для принятия решений?
5. На что стоит обратить внимание при интерпретации?

Объясни простым языком, избегая сложной статистической терминологии.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Ошибка GPT анализа корреляций: {e}")
            return f"Найдено {len(correlations)} корреляций. Детальный анализ недоступен."

    def generate_executive_summary(self,
                                   analysis_results: List[Dict],
                                   table_summary: Dict) -> str:
        """Генерирует executive summary с помощью GPT"""

        prompt = f"""
На основе проведенного анализа данных создай executive summary для руководства.

Результаты анализа:
{json.dumps(analysis_results, indent=2, ensure_ascii=False)}

Общая информация:
{json.dumps(table_summary, indent=2, ensure_ascii=False)}

ЗАДАЧА: Создай краткое (3-4 абзаца) executive summary, которое включает:
1. Ключевые находки из анализа данных
2. Самые важные инсайты для бизнеса
3. Критические проблемы, требующие внимания
4. Топ-3 рекомендации для действий

Пиши для руководителей, которые принимают стратегические решения.
Фокусируйся на бизнес-ценности, а не на технических деталях.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=800
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Ошибка создания executive summary: {e}")
            return "Executive summary временно недоступен."
