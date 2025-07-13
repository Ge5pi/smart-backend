# services/agents.py
import ast
import json
import io
import re
import uuid
import pandas as pd
import openai
from sqlalchemy import inspect, create_engine

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
import plotly.express as px

import config
import boto3

s3_client = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_DEFAULT_REGION
)


# --- Агент 1 и 2 (без изменений) ---

def get_schema_details(engine):
    inspector = inspect(engine)
    schema_info = []
    table_names = inspector.get_table_names()
    for table_name in table_names:
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        schema_info.append(f"Table '{table_name}' with columns: {columns}")
    return "\n".join(schema_info) if schema_info else "No tables found."


def create_analysis_plan(schema_details: str, client: openai.OpenAI):
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
    - **Include at least ONE question about trends over time** (e.g., using `datetime_created`).
    - **Include at least ONE question that requires correlating/joining data from two different tables** (e.g., 'users' and 'an_files').
    - **Include at least ONE question about finding distributions or anomalies** (e.g., 'What is the distribution of report statuses?', 'Are there users with an unusually high number of uploads?').

    Return the plan as a single JSON array of strings.

    Database Schema:
    {schema_details}
    """
    try:
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        result_data = json.loads(response.choices[0].message.content)
        if isinstance(result_data, dict):
            for key, value in result_data.items():
                if isinstance(value, list):
                    return value
        elif isinstance(result_data, list):
            return result_data
        raise ValueError("The model did not return a list in the JSON response.")
    except Exception as e:
        print(f"Error creating analysis plan: {e}")
        return ["Count the number of rows in each table."]


# --- ИЗМЕНЕНИЯ ЗДЕСЬ ---

# -- Агент 3: SQL-аналитик (Супер-надежный) --
def run_sql_query_agent(engine, question: str) -> tuple[pd.DataFrame, str]:
    """
    Выполняет SQL-запрос с улучшенным промптом, который нацеливает
    агента на более глубокий анализ.
    """
    try:
        db = SQLDatabase(engine=engine)
        llm = ChatOpenAI(model="o4-mini", temperature=0.2)  # Слегка повысим креативность

        # --- УЛУЧШЕННЫЙ ПРОМПТ ДЛЯ АГЕНТА ---
        agent_prompt_prefix = """
        You are an expert data analyst who is a master of SQL. Your task is to answer the user's question about the database.
        - First, think step-by-step to understand the user's intent.
        - If a question requires joining data from multiple tables to find a correlation, you must write a query with a JOIN.
        - When dealing with dates, use appropriate date functions.
        - After executing the query, do not just return the data. Your final answer should be a concise summary of the findings.
        - Always double-check your query before execution.
        """

        agent_executor = create_sql_agent(
            llm,
            db=db,
            agent_type="openai-tools",
            verbose=True,
            prefix=agent_prompt_prefix  # <-- Добавляем нашу новую инструкцию
        )

        result = agent_executor.invoke({"input": question})
        final_output_text = result.get('output', '')

        # --- НАДЕЖНЫЙ ПАРСЕР (без изменений) ---
        # Стратегия 1: Промежуточные шаги
        if 'intermediate_steps' in result and result['intermediate_steps']:
            raw_data = result['intermediate_steps'][-1][1]
            if isinstance(raw_data, str) and re.search(r"\[\s*\(.*?\)\s*\]", raw_data, re.DOTALL):
                try:
                    data_list = ast.literal_eval(re.search(r"(\[.*\])", raw_data, re.DOTALL).group(1))
                    if data_list: return pd.DataFrame(data_list), final_output_text
                except Exception:
                    pass
            elif isinstance(raw_data, list) and raw_data:
                return pd.DataFrame(raw_data), final_output_text
        # Стратегия 2: Финальный ответ
        match = re.search(r"\[\s*\(.*?\)\s*\]", final_output_text, re.DOTALL)
        if match:
            try:
                data_list = ast.literal_eval(match.group(0))
                if data_list: return pd.DataFrame(data_list), final_output_text
            except Exception:
                pass

        return pd.DataFrame(), final_output_text

    except Exception as e:
        error_message = f"Критическая ошибка в SQL агенте: {e}"
        return pd.DataFrame(), error_message

    except Exception as e:
        error_message = f"Критическая ошибка в SQL агенте для вопроса '{question}': {e}"
        print(error_message)
        return pd.DataFrame(), error_message


# -- Агент 4: Визуализатор (без изменений) ---
def create_visualization(df: pd.DataFrame, question: str) -> str | None:
    if df.empty or len(df.columns) < 2:
        return None
    try:
        if 'date' in df.columns[0].lower() or 'month' in df.columns[0].lower():
            chart_type = 'line'
        else:
            chart_type = 'bar'
        fig = None
        x_col, y_col = df.columns[0], df.columns[1]
        if chart_type == 'bar':
            fig = px.bar(df, x=x_col, y=y_col, title=question)
        elif chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=question)
        if fig:
            img_bytes = io.BytesIO()
            fig.write_image(img_bytes, format="png")
            img_bytes.seek(0)
            file_name = f"charts/{uuid.uuid4()}.png"
            s3_client.upload_fileobj(img_bytes, config.S3_BUCKET_NAME, file_name,
                                     ExtraArgs={'ContentType': 'image/png'})
            return f"https://{config.S3_BUCKET_NAME}.s3.{config.AWS_DEFAULT_REGION}.amazonaws.com/{file_name}"
        return None
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None


# -- Агент 5: Интерпретатор (улучшенный) --
def create_narrative(question: str, df: pd.DataFrame, chart_url: str | None, raw_text: str, client: openai.OpenAI):
    """
    Генерирует описание. Если DataFrame пустой, но есть сырой текст от агента,
    анализирует этот текст.
    """
    # Сценарий 1: У нас есть данные в DataFrame
    if not df.empty:
        df_markdown = df.head(10).to_markdown()
        prompt = f"""
        You are a data analyst. Your task is to provide a clear narrative based on the data provided.
        Original Question: {question}
        Data Summary (Markdown):
        {df_markdown}
        A chart is available at: {chart_url if chart_url else "No chart."}
        Your Narrative (summarize findings and explain the chart):
        """
    # Сценарий 2: Данных нет, но есть текстовый ответ от предыдущего агента
    elif raw_text:
        prompt = f"""
        You are a data analyst. A previous AI agent failed to return structured data for the question "{question}".
        However, it provided the following text. Your task is to analyze this text, summarize it for the user,
        and explain what might have happened. If it looks like an error message, explain it in simple terms.
        Agent's raw text response:
        ---
        {raw_text}
        ---
        Your summary and analysis for the user:
        """
    # Сценарий 3: Нет ни данных, ни текста
    else:
        return "The query did not return any data or text. Unable to provide analysis."

    try:
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error creating narrative: {e}")
        return "Failed to generate a narrative for the findings."
