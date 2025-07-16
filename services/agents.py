# agents.py
import ast
import io
import json
import re
import uuid

import boto3
import openai
import pandas as pd
import plotly.express as px
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import inspect

import config

# --- Клиенты и утилиты ---
s3_client = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_DEFAULT_REGION
)
openai_client = openai.OpenAI(api_key=config.API_KEY)


def create_visualization(df: pd.DataFrame, question: str) -> str | None:
    """Создает визуализацию и загружает в S3, возвращая публичную ссылку."""
    if df.empty or len(df.columns) < 2 or len(df) < 2:
        return None
    try:
        x_col, y_col = df.columns[0], df.columns[1]
        x_is_date = pd.api.types.is_datetime64_any_dtype(df[x_col])
        y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])

        fig = None
        if x_is_date and y_is_numeric:
            fig = px.line(df, x=x_col, y=y_col, title=question, markers=True)
        elif y_is_numeric:
            fig = px.bar(df.sort_values(by=y_col, ascending=False).head(20), x=x_col, y=y_col, title=question)

        if fig:
            img_bytes = io.BytesIO()
            fig.write_image(img_bytes, format="png", scale=2)
            img_bytes.seek(0)
            file_name = f"charts/{uuid.uuid4()}.png"
            s3_client.upload_fileobj(
                img_bytes, config.S3_BUCKET_NAME, file_name,
                ExtraArgs={'ContentType': 'image/png', 'ACL': 'public-read'}
            )
            return f"https://{config.S3_BUCKET_NAME}.s3.{config.AWS_DEFAULT_REGION}.amazonaws.com/{file_name}"
        return None
    except Exception as e:
        print(f"Error creating visualization for question '{question}': {e}")
        return None


# --- Базовый класс агента ---
class BaseAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.client = openai_client
        self.model = model


# --- 1. Агент-Оркестратор ---
class Orchestrator(BaseAgent):
    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine

    def get_schema_details(self):
        inspector = inspect(self.engine)
        schema_info = []
        for table_name in inspector.get_table_names():
            columns = [f"{col['name']} ({col['type']})" for col in inspector.get_columns(table_name)]
            schema_info.append(f"Table '{table_name}' with columns: {columns}")
        return "\n".join(schema_info) or "No tables found."

    def create_initial_plan(self):
        schema = self.get_schema_details()
        prompt = f"""
        You are a principal data analyst. Your goal is to uncover deep, non-obvious, actionable insights.
        Based on the provided schema, create a strategic analysis plan as a JSON array of 5-7 probing questions.
        Focus on correlations, trends, anomalies, and user segmentation. Avoid simple counts.

        **MUST-INCLUDE Question Types:**
        - Time-Series/Trend Analysis.
        - Cross-Table Correlation (requiring a JOIN).
        - Distribution/Anomaly Detection.

        Return ONLY the JSON array of strings.

        **Database Schema:**
        ```
        {schema}
        ```
        """
        response = self.client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.2
        )
        try:
            data = json.loads(response.choices[0].message.content)
            for key, value in data.items():
                if isinstance(value, list): return value
            return ["Analyze the main table."]
        except Exception as e:
            print(f"Error decoding initial plan, using fallback: {e}")
            return ["Count rows in all tables.", "Describe the schema."]

    def process_evaluation(self, evaluation: dict, session_memory: list, analysis_plan: list):
        session_memory.append(evaluation['finding'])
        if evaluation.get('new_hypotheses'):
            analysis_plan[:0] = evaluation['new_hypotheses']


# --- 2. Агент-Исполнитель SQL ---
class SQLCoder(BaseAgent):
    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.db = SQLDatabase(engine=engine)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=config.API_KEY)

    def run(self, question: str) -> dict:
        prefix = """
        You are a master SQL data analyst. Your task is to write a syntactically correct SQL query to answer the user's question.
        Think step-by-step. Double-check table and column names. Use date functions for time-based questions.
        The final answer must ONLY be the result of the query execution.
        """
        agent_executor = create_sql_agent(llm=self.llm, db=self.db, agent_type="openai-tools", verbose=False, prefix=prefix, handle_parsing_errors=True)
        try:
            result = agent_executor.invoke({"input": question})
            raw_output = result.get('output', '')
            df = pd.DataFrame()

            if 'intermediate_steps' in result and result['intermediate_steps']:
                # Более надежный парсинг из вывода LangChain
                query = result['intermediate_steps'][0][1].get('query', result['intermediate_steps'][0][1])
                with self.db._engine.connect() as connection:
                    df = pd.read_sql(query, connection)

            return {"question": question, "data": df, "raw_output": raw_output, "error": None}
        except Exception as e:
            print(f"CRITICAL ERROR in SQLCoder for question '{question}': {e}")
            return {"question": question, "data": pd.DataFrame(), "raw_output": f"An error occurred: {e}", "error": str(e)}


# --- 3. Агент-Критик ---
class Critic(BaseAgent):
    def evaluate(self, execution_result: dict):
        question = execution_result['question']
        df = execution_result['data']
        if execution_result['error']:
            return {"is_success": False, "finding": {"question": question, "summary": f"Query failed: {execution_result['error']}"}}
        if df.empty:
            return {"is_success": True, "finding": {"question": question, "summary": "Query executed but returned no data.", "chart_url": None, "data_preview": None}, "new_hypotheses": []}

        chart_url = create_visualization(df, question)
        prompt = f"""
        You are a Senior Data Analyst and Insight Generator. Analyze the result of a SQL query, extract meaningful insights, and brainstorm new questions to dig deeper.

        **Analysis Context:**
        - **Original Question:** "{question}"
        - **Data Result (first 10 rows):**
        ```
        {df.head(10).to_markdown(index=False)}
        ```
        - **Data Shape:** {df.shape[0]} rows, {df.shape[1]} columns.
        - **Visualization:** A chart is available at: {chart_url or "No chart."}

        **Your Tasks (Respond in valid JSON format):**
        1.  **`summary` (string):** Write a concise, insightful summary of the findings. Interpret the data, don't just state it. Mention the chart.
        2.  **`new_hypotheses` (array of strings):** Generate 1-3 new, intelligent follow-up questions to explore *why* the data looks this way.
            - Bad hypothesis: "Show me the data again".
            - Good hypothesis: "The spike in June registrations is interesting. Let's analyze the `user_id`s created that month to see if it's from a single source."

        **OUTPUT MUST BE A VALID JSON OBJECT.**
        """
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        try:
            eval_data = json.loads(response.choices[0].message.content)
            return {
                "is_success": True,
                "finding": {"question": question, "summary": eval_data.get("summary"), "chart_url": chart_url, "data_preview": df.head().to_dict('records')},
                "new_hypotheses": eval_data.get("new_hypotheses", [])
            }
        except Exception as e:
            return {"is_success": False, "finding": {"question": question, "summary": f"Failed to parse critic evaluation: {e}"}}


# --- 4. Агент-Синтезатор ---
class Storyteller(BaseAgent):
    def narrate(self, session_memory: list):
        findings_text = "\n\n".join([f"### {idx+1}. On: '{finding['question']}'\n**Summary:** {finding['summary']}" for idx, finding in enumerate(session_memory)])
        prompt = f"""
        You are a Lead Data Analyst presenting to an executive. Synthesize the following series of findings into a cohesive report with an executive summary and detailed points.

        **Analytical Findings:**
        ---
        {findings_text}
        ---

        **Your Output (JSON Object):**
        Generate a JSON with two keys: `executive_summary` and `detailed_findings`.
        1.  **`executive_summary` (string):** High-level, 3-4 paragraph summary. Start with the most impactful conclusion and recommendations.
        2.  **`detailed_findings` (array of objects):** Reuse the provided info. Each object should have keys: `question`, `summary`, `chart_url`, `data_preview`.

        **Tone:** Confident, insightful, business-oriented. Focus on the 'so what'. Respond ONLY with the JSON.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        try:
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"executive_summary": "Failed to generate the final report.", "detailed_findings": session_memory}