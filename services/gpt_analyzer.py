# services/smart_gpt_analyzer.py - ПОЛНОСТЬЮ ПЕРЕРАБОТАННЫЙ
import openai
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import json
from config import API_KEY
import re

logger = logging.getLogger(__name__)


class SmartGPTAnalyzer:
    """Умный GPT-аналитик, который интерпретирует НАЙДЕННЫЕ паттерны"""

    def __init__(self):
        self.client = openai.OpenAI(api_key=API_KEY)
        self.model = "gpt-4o-mini"

    def analyze_findings_with_context(self,
                                      dataframe_results: Dict[str, Any],
                                      business_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Анализирует НАЙДЕННЫЕ паттерны и дает бизнес-интерпретацию"""

        # Определяем бизнес-контекст из данных
        auto_context = self._detect_business_context(dataframe_results)
        context = {**auto_context, **(business_context or {})}

        # Создаем структурированный анализ найденных паттернов
        prompt = self._create_smart_analysis_prompt(dataframe_results, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_smart_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Более детерминированный
                max_tokens=1500
            )

            analysis = response.choices[0].message.content

            # Парсим структурированный ответ
            parsed_analysis = self._parse_gpt_response(analysis)

            return {
                "business_insights": parsed_analysis.get("insights", analysis),
                "action_items": parsed_analysis.get("actions", []),
                "risk_assessment": parsed_analysis.get("risks", ""),
                "opportunities": parsed_analysis.get("opportunities", []),
                "confidence": "high",
                "business_context": context
            }

        except Exception as e:
            logger.error(f"Ошибка умного GPT анализа: {e}")
            return {
                "business_insights": f"Технический анализ завершен, но GPT-интерпретация недоступна: {str(e)}",
                "confidence": "low"
            }

    def _detect_business_context(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Автоматически определяет бизнес-контекст из названий колонок и данных"""

        analyzed_tables = results.get('analyzed_tables', [])
        data_preview = results.get('data_preview', [])

        context = {
            "domain": "general",
            "key_metrics": [],
            "entities": [],
            "time_dimension": False
        }

        # Анализируем названия колонок для определения домена
        all_columns = []
        if data_preview:
            all_columns = list(data_preview[0].keys()) if data_preview else []

        # Определяем домен бизнеса
        column_text = " ".join(all_columns).lower()

        if any(word in column_text for word in ['price', 'cost', 'revenue', 'profit', 'amount', 'payment']):
            context["domain"] = "financial"
            context["key_metrics"] = ["revenue", "cost", "profit_margin"]
        elif any(word in column_text for word in ['user', 'customer', 'client', 'visitor']):
            context["domain"] = "customer"
            context["key_metrics"] = ["acquisition", "retention", "churn"]
        elif any(word in column_text for word in ['sale', 'order', 'product', 'item', 'quantity']):
            context["domain"] = "sales"
            context["key_metrics"] = ["volume", "conversion", "average_order_value"]
        elif any(word in column_text for word in ['employee', 'staff', 'hr', 'salary']):
            context["domain"] = "hr"
            context["key_metrics"] = ["headcount", "turnover", "productivity"]
        elif any(word in column_text for word in ['inventory', 'stock', 'supply', 'warehouse']):
            context["domain"] = "operations"
            context["key_metrics"] = ["efficiency", "utilization", "turnover"]

        # Проверяем наличие временных данных
        if any(word in column_text for word in ['date', 'time', 'year', 'month', 'created', 'updated']):
            context["time_dimension"] = True

        # Извлекаем основные сущности
        context["entities"] = analyzed_tables

        return context

    def _create_smart_analysis_prompt(self, results: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Создает умный промпт на основе найденных паттернов"""

        summary = results.get('summary', '')
        basic_info = results.get('basic_info', {})
        anomalies = results.get('anomalies', [])
        correlations = results.get('correlations', [])
        data_preview = results.get('data_preview', [])[:3]  # Только первые 3 записи

        domain_context = {
            "financial": "Это финансовые данные. Фокусируйся на прибыльности, рисках, трендах доходов.",
            "customer": "Это данные о клиентах. Анализируй поведение, сегменты, лояльность.",
            "sales": "Это данные продаж. Изучай конверсию, сезонность, эффективность каналов.",
            "hr": "Это HR данные. Анализируй продуктивность, удержание, затраты на персонал.",
            "operations": "Это операционные данные. Фокусируйся на эффективности и оптимизации."
        }.get(context["domain"], "Это бизнес-данные. Ищи возможности для роста и оптимизации.")

        prompt = f"""
КОНТЕКСТ: {domain_context}

ТЕХНИЧЕСКИЕ РЕЗУЛЬТАТЫ АНАЛИЗА:
{summary}

КЛЮЧЕВАЯ СТАТИСТИКА:
- Записей: {basic_info.get('rows', 'N/A')}
- Параметров: {basic_info.get('columns', 'N/A')}
- Объем данных: {basic_info.get('memory_mb', 'N/A')} MB

НАЙДЕННЫЕ АНОМАЛИИ:
{json.dumps(anomalies, indent=2, ensure_ascii=False) if anomalies else "Аномалии не обнаружены"}

НАЙДЕННЫЕ КОРРЕЛЯЦИИ:
{json.dumps(correlations, indent=2, ensure_ascii=False) if correlations else "Корреляции не найдены"}

ПРИМЕРЫ ДАННЫХ:
{json.dumps(data_preview, indent=2, ensure_ascii=False)}

ЗАДАЧА: 
На основе этих КОНКРЕТНЫХ технических находок дай ПРАКТИЧЕСКИЙ бизнес-анализ:

[INSIGHTS]
3-4 конкретных инсайта на основе найденных паттернов

[ACTIONS] 
Конкретные действия, которые нужно предпринять

[RISKS]
Критические риски из найденных аномалий

[OPPORTUNITIES]
2-3 возможности для роста/оптимизации

НЕ придумывай данные. Анализируй только то, что НАЙДЕНО техническим анализом.
"""
        return prompt

    def _get_smart_system_prompt(self) -> str:
        """Системный промпт для умного анализа"""
        return """Ты - старший бизнес-аналитик с 10+ лет опыта в data science.

ТВОЯ ЗАДАЧА: Интерпретировать результаты технического анализа данных и давать КОНКРЕТНЫЕ бизнес-рекомендации.

ПРАВИЛА:
1. Анализируй ТОЛЬКО найденные техническим анализом паттерны
2. НЕ придумывай данные, которых нет
3. Давай КОНКРЕТНЫЕ, actionable рекомендации
4. Фокусируйся на БИЗНЕС-ЦЕННОСТИ находок
5. Указывай ПРИОРИТЕТЫ действий
6. Оценивай РИСКИ и ВОЗМОЖНОСТИ

ФОРМАТ ОТВЕТА:
[INSIGHTS] - ключевые бизнес-находки
[ACTIONS] - что делать прямо сейчас  
[RISKS] - на что обратить внимание
[OPPORTUNITIES] - где искать рост

Пиши по-русски, кратко, по делу."""

    def _parse_gpt_response(self, response: str) -> Dict[str, Any]:
        """Парсит структурированный ответ GPT"""
        parsed = {}

        try:
            # Извлекаем секции
            insights_match = re.search(r'\[INSIGHTS\](.*?)\[ACTIONS\]', response, re.DOTALL)
            actions_match = re.search(r'\[ACTIONS\](.*?)\[RISKS\]', response, re.DOTALL)
            risks_match = re.search(r'\[RISKS\](.*?)\[OPPORTUNITIES\]', response, re.DOTALL)
            opportunities_match = re.search(r'\[OPPORTUNITIES\](.*?)$', response, re.DOTALL)

            parsed["insights"] = insights_match.group(1).strip() if insights_match else ""
            parsed["actions"] = self._extract_list_items(actions_match.group(1).strip() if actions_match else "")
            parsed["risks"] = risks_match.group(1).strip() if risks_match else ""
            parsed["opportunities"] = self._extract_list_items(
                opportunities_match.group(1).strip() if opportunities_match else "")

        except Exception as e:
            logger.error(f"Ошибка парсинга GPT ответа: {e}")
            parsed["insights"] = response
            parsed["actions"] = []
            parsed["risks"] = ""
            parsed["opportunities"] = []

        return parsed

    def _extract_list_items(self, text: str) -> List[str]:
        """Извлекает элементы списка из текста"""
        if not text:
            return []

        # Разделяем по переносам и фильтруем
        items = []
        for line in text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('1.') or line.startswith(
                    '2.') or line.startswith('3.')):
                # Убираем маркеры
                clean_line = re.sub(r'^[-•\d\.]\s*', '', line)
                if clean_line:
                    items.append(clean_line)

        return items

    def generate_executive_summary_smart(self,
                                         all_findings: List[Dict[str, Any]],
                                         table_summary: Dict[str, Any]) -> str:
        """Генерирует умное executive summary"""

        # Собираем все бизнес-инсайты
        business_insights = []
        action_items = []
        risks = []
        opportunities = []

        for finding in all_findings:
            gpt_data = finding.get('gpt_insights', {})
            if gpt_data:
                if gpt_data.get('business_insights'):
                    business_insights.append(gpt_data['business_insights'])
                if gpt_data.get('action_items'):
                    action_items.extend(gpt_data['action_items'])
                if gpt_data.get('risk_assessment'):
                    risks.append(gpt_data['risk_assessment'])
                if gpt_data.get('opportunities'):
                    opportunities.extend(gpt_data['opportunities'])

        if not business_insights:
            return "Анализ данных завершен. Детальные инсайты формируются..."

        # Создаем промпт для executive summary
        prompt = f"""
На основе проведенного анализа создай executive summary для руководства:

КЛЮЧЕВЫЕ НАХОДКИ:
{' '.join(business_insights[:3])}

КРИТИЧНЫЕ ДЕЙСТВИЯ:
{' '.join(action_items[:5])}

РИСКИ:
{' '.join(risks[:3])}

ВОЗМОЖНОСТИ:
{' '.join(opportunities[:3])}

ОБЩАЯ СТАТИСТИКА:
- Таблиц проанализировано: {table_summary.get('total_tables', 0)}
- Связей найдено: {table_summary.get('total_relations', 0)}
- Объем данных: {table_summary.get('total_memory_mb', 0):.1f} MB

Создай краткое (2-3 абзаца) executive summary для руководителя, который принимает решения.
Фокусируйся на БИЗНЕС-ЦЕННОСТИ и ПРИОРИТЕТНЫХ ДЕЙСТВИЯХ.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "Ты создаешь executive summary для CEO. Пиши кратко, четко, акцент на ROI и действиях."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Ошибка создания умного executive summary: {e}")
            return f"Проведен комплексный анализ {table_summary.get('total_tables', 0)} таблиц данных. Выявлены ключевые паттерны и возможности для оптимизации. Детали в разделах анализа."
