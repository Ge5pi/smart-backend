# services/agents.py
import ast
import json
import io
import re
import uuid
import logging
from typing import List, Dict, Any, Tuple, Optional, Union

import pandas as pd
import openai
from sqlalchemy import inspect, create_engine
from sqlalchemy.engine import Engine

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
import plotly.express as px
import plotly.io as pio

# Local imports
import config
import boto3

# --- ИЗМЕНЕНИЕ: Настройка логирования ---
# Заменяет все print() на структурированное логирование.
# Ошибки будут писаться в консоль/файл с полной информацией,
# а пользователь увидит только безопасное сообщение.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ИЗМЕНЕНИЕ: Централизованные константы ---
# Упрощает замену моделей и управление параметрами
O4_MINI_MODEL = "o4-mini"

# --- Клиент S3 (без изменений) ---
s3_client = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_DEFAULT_REGION
)


# --- Агент 1: Получение схемы (без существенных изменений) ---
def get_schema_details(engine: Engine) -> str:
    """Возвращает строковое представление схемы базы данных."""
    try:
        inspector = inspect(engine)
        schema_info = []
        table_names = inspector.get_table_names()
        for table_name in table_names:
            columns = [col['name'] for col in inspector.get_columns(table_name)]
            schema_info.append(f"Table '{table_name}' with columns: {columns}")
        return "\n".join(schema_info) if schema_info else "No tables found in the database."
    except Exception as e:
        logger.error(f"Failed to get schema details: {e}", exc_info=True)
        return "Error: Could not retrieve database schema."


# --- Агент 2: Планировщик анализа (улучшено логирование и обработка JSON) ---
def create_analysis_plan(schema_details: str, client: openai.OpenAI) -> List[str]:
    """
    Создает план анализа, фокусируясь на поиске неочевидных инсайтов,
    корреляций и аномалий.
    """
    prompt = f"""
    You are an expert data analyst. Your goal is to find **interesting and non-obvious insights**
    hidden in the data. Based on the database schema provided, create a step-by-step analysis plan.

    **Instructions for the plan:**
    - The plan must consist of 5-7 analytical questions.
    - Questions should be complex and aim to uncover relationships, not just state simple facts.
    - **Include at least ONE question about trends over time.**
    - **Include at least ONE question that requires correlating data from two different tables.**
    - **Include at least ONE question about finding distributions or anomalies.**

    Return the plan as a single JSON array of strings, like this: ["Question 1", "Question 2", ...].

    Database Schema:
    {schema_details}
    """
    try:
        response = client.chat.completions.create(
            model=O4_MINI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        # --- ИЗМЕНЕНИЕ: Более надежный парсинг JSON ---
        content = response.choices[0].message.content
        result_data = json.loads(content)

        # Модель может вернуть {"plan": [...]} или просто [...]
        if isinstance(result_data, dict):
            # Ищем первый ключ, значение которого является списком
            for key, value in result_data.items():
                if isinstance(value, list):
                    return value
        elif isinstance(result_data, list):
            return result_data

        raise ValueError("The model did not return a list in the JSON response.")
    except Exception as e:
        # --- ИЗМЕНЕНИЕ: Логируем ошибку, а не выводим ее ---
        logger.error(f"Error creating analysis plan: {e}\nRaw response: {content}", exc_info=True)
        # Возвращаем безопасный план по умолчанию
        return ["Count the number of rows in each table.", "Describe the main tables."]


# --- Агент 3: SQL-аналитик (значительно переработан) ---
def run_sql_query_agent(engine: Engine, question: str) -> Tuple[pd.DataFrame, str]:
    """
    Выполняет SQL-запрос с помощью LangChain агента.
    Улучшена обработка ошибок и парсинг результата.
    """
    try:
        db = SQLDatabase(engine=engine)
        llm = ChatOpenAI(model=O4_MINI_MODEL, temperature=0) # Убрали "креативность" для точности SQL

        # --- ИЗМЕНЕНИЕ: Слегка улучшенный промпт ---
        agent_prompt_prefix = """
        You are an expert SQL data analyst. Your task is to answer the user's question about the database.
        - First, think step-by-step to understand the user's intent.
        - Write a syntactically correct SQL query for the user's question.
        - If a question requires joining data, you must write a query with a JOIN.
        - When dealing with dates, use appropriate date functions.
        - After executing the query, do not just return the data. Your final answer should be a concise summary of the findings.
        - Always double-check your query before execution.
        """

        agent_executor = create_sql_agent(
            llm,
            db=db,
            agent_type="openai-tools",
            verbose=False, # Отключаем verbose в продакшене, чтобы не засорять логи
            prefix=agent_prompt_prefix
        )

        result = agent_executor.invoke({"input": question})
        final_output_text = result.get('output', '')

        # --- ИЗМЕНЕНИЕ: Более надежная стратегия парсинга ---
        raw_data = None
        # Стратегия 1: Искать в промежуточных шагах (самый надежный источник)
        if 'intermediate_steps' in result and result['intermediate_steps']:
            # Данные обычно в последнем шаге
            raw_data = result['intermediate_steps'][-1][1]

        # Стратегия 2: Если в шагах нет, искать в финальном тексте
        if not raw_data:
            raw_data = final_output_text

        if isinstance(raw_data, list) and raw_data:
            # Если это уже готовый список словарей или кортежей
            return pd.DataFrame(raw_data), final_output_text
        elif isinstance(raw_data, str):
            # Если это строка, пытаемся извлечь из нее список
            # Регулярное выражение для поиска [...]
            match = re.search(r"\[\s*\(.*?\)\s*\]|\[\s*\{.*?\}\s*\]", raw_data, re.DOTALL)
            if match:
                try:
                    # Используем безопасный ast.literal_eval для преобразования строки в Python объект
                    data_list = ast.literal_eval(match.group(0))
                    if data_list:
                        return pd.DataFrame(data_list), final_output_text
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"Could not parse string to list: {e}. Raw string part: {match.group(0)}")

        # Если ничего не удалось распарсить, возвращаем пустой DataFrame, но с текстом
        return pd.DataFrame(), final_output_text

    except Exception as e:
        # --- ИЗМЕНЕНИЕ: Логируем полную ошибку для отладки ---
        logger.error(f"Critical error in SQL agent for question '{question}': {e}", exc_info=True)
        # --- А пользователю возвращаем простое сообщение ---
        error_message = f"Произошла критическая ошибка при выполнении SQL-запроса. Анализ не может быть продолжен."
        return pd.DataFrame(), error_message


# --- Агент 4: Визуализатор (улучшенная логика выбора графика) ---
def create_visualization(df: pd.DataFrame, question: str) -> Optional[str]:
    """
    Создает визуализацию на основе DataFrame и загружает на S3.
    Автоматически выбирает тип графика на основе типов данных.
    """
    if df.empty or len(df.columns) < 2:
        return None

    try:
        # --- ИЗМЕНЕНИЕ: Умное определение типа графика ---
        x_col, y_col = df.columns[0], df.columns[1]
        fig = None

        # Проверяем типы данных для более точного выбора графика
        is_x_date = pd.api.types.is_datetime64_any_dtype(df[x_col])
        is_x_numeric = pd.api.types.is_numeric_dtype(df[x_col])
        is_y_numeric = pd.api.types.is_numeric_dtype(df[y_col])

        if is_x_date and is_y_numeric:
            # Тренд по времени -> линейный график
            fig = px.line(df, x=x_col, y=y_col, title=question, markers=True)
        elif is_x_numeric and is_y_numeric:
            # Корреляция двух чисел -> диаграмма рассеяния
            fig = px.scatter(df, x=x_col, y=y_col, title=question)
        else:
            # Категориальные данные -> столбчатая диаграмма (по умолчанию)
            fig = px.bar(df, x=x_col, y=y_col, title=question)

        if fig:
            img_bytes = io.BytesIO()
            pio.write_image(img_bytes, fig, format="png") # Используем pio для большей надежности
            img_bytes.seek(0)

            file_name = f"charts/{uuid.uuid4()}.png"
            s3_client.upload_fileobj(
                img_bytes,
                config.S3_BUCKET_NAME,
                file_name,
                ExtraArgs={'ContentType': 'image/png', 'ACL': 'public-read'} # Делаем файл публичным
            )
            # --- ИЗМЕНЕНИЕ: Формирование URL-адреса, совместимого с разными регионами ---
            return f"https://{config.S3_BUCKET_NAME}.s3.amazonaws.com/{file_name}"

        return None
    except Exception as e:
        logger.error(f"Error creating visualization: {e}", exc_info=True)
        return None


# --- Агент 5: Интерпретатор (улучшен промпт для обработки ошибок) ---
def create_narrative(
    question: str,
    df: pd.DataFrame,
    chart_url: Optional[str],
    raw_text: str,
    client: openai.OpenAI
) -> str:
    """
    Генерирует человекочитаемое описание результатов или ошибки.
    """
    prompt = ""
    # Сценарий 1: Есть данные
    if not df.empty:
        df_markdown = df.head(10).to_markdown(index=False)
        prompt = f"""
        As a data analyst, provide a clear, concise narrative based on the provided data.

        **Original Question:** {question}

        **Data Summary (first 10 rows):**
        ```markdown
        {df_markdown}
        ```

        **Visualization:** A {'line chart' if chart_url and 'line' in chart_url else 'bar chart'} is available at: {chart_url if chart_url else "No chart was generated."}

        **Your Narrative:**
        - Start with a direct answer to the question.
        - Briefly explain the findings shown in the data and the chart.
        - Conclude with a key insight or takeaway.
        """
    # Сценарий 2: Данных нет, но есть текстовый ответ (возможно, ошибка)
    elif raw_text:
        # --- ИЗМЕНЕНИЕ: Улучшенный промпт, чтобы скрыть технические детали от пользователя ---
        prompt = f"""
        You are a helpful assistant. An attempt to answer the question "{question}" failed to produce structured data.
        The system returned the following text.

        **Your Task:**
        1. Analyze the text below.
        2. **Do not show the raw text to the user.**
        3. Explain in simple, non-technical terms what happened. If it's an error, suggest what might have gone wrong (e.g., "The query could not be completed," or "The data for this question may not exist").
        4. Provide a friendly, helpful message to the user.

        **Agent's Raw Text:**
        ---
        {raw_text}
        ---

        **Your Summary for the User:**
        """
    # Сценарий 3: Нет ни данных, ни текста
    else:
        return "The analysis did not return any data or text. It's not possible to provide a summary."

    try:
        response = client.chat.completions.create(
            model=O4_MINI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error creating narrative: {e}", exc_info=True)
        return "Failed to generate a narrative for the findings."