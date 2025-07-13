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

# S3 Client and config from main.py (we need to find a way to share it)
# For now, let's assume we can import them or re-initialize
import config  # Assuming you have S3 config here
import boto3

s3_client = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_DEFAULT_REGION
)


# -- Агент 1: Исследователь схемы --
def get_schema_details(engine):
    """Подключается к БД и извлекает ее схему в текстовом виде."""
    inspector = inspect(engine)
    schema_info = []
    table_names = inspector.get_table_names()
    for table_name in table_names:
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        schema_info.append(f"Table '{table_name}' with columns: {columns}")
    return "\n".join(schema_info) if schema_info else "No tables found."


# -- Агент 2: Главный аналитик (Планировщик) --
def create_analysis_plan(schema_details: str, client: openai.OpenAI):
    """На основе схемы БД создает JSON-план анализа."""
    prompt = f"""
    You are a principal data analyst. Based on the database schema provided, create a concise,
    step-by-step analysis plan. The plan should consist of 5-7 key business questions that can be answered using this data.
    Return the plan as a JSON array of strings.

    Database Schema:
    {schema_details}

    Example Output:
    ["Analyze sales dynamics by month", "Identify the top 10 most profitable products", "Segment users by purchase frequency"]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or your preferred model
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        # The model might return a JSON object with a key, e.g. {"plan": [...]}. We need to find the list.
        result_data = json.loads(response.choices[0].message.content)
        if isinstance(result_data, dict):
            for key, value in result_data.items():
                if isinstance(value, list):
                    return value  # Return the first list found in the JSON object
        elif isinstance(result_data, list):
            return result_data
        raise ValueError("The model did not return a list in the JSON response.")
    except Exception as e:
        print(f"Error creating analysis plan: {e}")
        # Fallback plan in case of error
        return ["Provide a general summary of each table."]


# -- Агент 3: SQL-аналитик (Исполнитель) --
def run_sql_query_agent(engine, question: str) -> pd.DataFrame:
    """
    Преобразует вопрос в SQL-запрос, выполняет его и возвращает результат как DataFrame,
    извлекая его напрямую из промежуточных шагов агента.
    """
    try:
        db = SQLDatabase(engine=engine)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

        # Вызываем агент
        result = agent_executor.invoke({"input": question})

        # --- ИСПРАВЛЕННАЯ ЛОГИКА ---
        # Ищем результат не в 'output', а в 'intermediate_steps'
        if 'intermediate_steps' in result and result['intermediate_steps']:
            # Наблюдение (observation) от последнего шага - это то, что вернул инструмент
            raw_data = result['intermediate_steps'][-1][1]

            # Данные могут быть строкой, которую нужно преобразовать в список
            if isinstance(raw_data, str):
                try:
                    # Используем ast.literal_eval для безопасного преобразования строки в список
                    data_list = ast.literal_eval(raw_data)
                except (ValueError, SyntaxError):
                    print(f"Ошибка парсинга строки с данными: {raw_data}")
                    return pd.DataFrame()
            elif isinstance(raw_data, list):
                data_list = raw_data
            else:
                data_list = []

            if data_list:
                # Пытаемся получить имена столбцов из запроса агента для красивого DataFrame
                try:
                    query = result['intermediate_steps'][-1][0].tool_input['query']
                    columns = [desc[0] for desc in create_engine(db.uri).execute(query).cursor.description]
                    return pd.DataFrame(data_list, columns=columns)
                except:
                    return pd.DataFrame(data_list) # Фоллбэк, если не удалось получить столбцы

        print(f"Не удалось извлечь данные из промежуточных шагов. Финальный ответ агента: {result.get('output')}")
        return pd.DataFrame()

    except Exception as e:
        print(f"Критическая ошибка в SQL агенте для вопроса '{question}': {e}")
        return pd.DataFrame()

# -- Агент 4: Визуализатор --
def create_visualization(df: pd.DataFrame, question: str) -> str | None:
    """Создает график на основе DataFrame, загружает в S3 и возвращает URL."""
    if df.empty or len(df.columns) < 2:
        return None  # Cannot create a meaningful chart

    try:
        # Simple logic to determine chart type
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
            s3_client.upload_fileobj(
                img_bytes,
                config.S3_BUCKET_NAME,
                file_name,
                ExtraArgs={'ContentType': 'image/png'}
            )
            # Construct the public URL
            return f"https://{config.S3_BUCKET_NAME}.s3.{config.AWS_DEFAULT_REGION}.amazonaws.com/{file_name}"

        return None
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None


# -- Агент 5: Интерпретатор --
def create_narrative(question: str, df: pd.DataFrame, chart_url: str | None, client: openai.OpenAI):
    """Генерирует текстовое описание результатов анализа."""
    if df.empty:
        return "The query did not return any data. Unable to provide analysis."

    df_markdown = df.head(10).to_markdown()  # Show top 10 rows

    prompt = f"""
    You are a data analyst reporting your findings.
    Your task is to provide a clear, concise narrative based on the data provided.

    Original Question: {question}

    Data Summary (in Markdown format):
    {df_markdown}

    A chart visualizing this data is available at: {chart_url if chart_url else "No chart was generated."}

    Your Narrative:
    - Start with a direct answer to the question.
    - Briefly explain the key insights discovered from the data.
    - If a chart is available, describe what it shows.
    - Conclude with a potential business implication or a next step.
    Keep the tone professional and easy to understand.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error creating narrative: {e}")
        return "Failed to generate a narrative for the findings."