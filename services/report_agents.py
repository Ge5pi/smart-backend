# report_agents.py
import ast
import io
import json
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
s3_client = boto3.client('s3', aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
                         region_name=config.AWS_DEFAULT_REGION)
openai_client = openai.OpenAI(api_key=config.API_KEY)


def create_visualization(df: pd.DataFrame, question: str) -> str | None:
    if df.empty or len(df.columns) < 2 or len(df) < 2: return None
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
            s3_client.upload_fileobj(img_bytes, config.S3_BUCKET_NAME, file_name, ExtraArgs={'ContentType': 'image/png', 'ACL': 'public-read'})
            return f"https://{config.S3_BUCKET_NAME}.s3.{config.AWS_DEFAULT_REGION}.amazonaws.com/{file_name}"
        return None
    except Exception as e:
        print(f"Error creating visualization for question '{question}': {e}")
        return None

class BaseAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.client = openai_client
        self.model = model

class Orchestrator(BaseAgent):
    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine

    def get_schema_details(self):
        inspector = inspect(self.engine)
        schema_info = []
        for table_name in inspector.get_table_names():
            if table_name == 'alembic_version': continue
            columns = [f"{col['name']} ({col['type']})" for col in inspector.get_columns(table_name)]
            schema_info.append(f"Table '{table_name}' with columns: {columns}")
        return "\n".join(schema_info) or "No tables found."

    def create_initial_plan(self):
        print("[Orchestrator:create_initial_plan] Generating plan...")
        schema = self.get_schema_details()
        prompt = f"""You are a principal data analyst. Your goal is to uncover deep, non-obvious, actionable 
        insights. Based on the provided schema, create a strategic analysis plan as a JSON array of 5-7 probing 
        questions. Focus on correlations, trends, anomalies, and user segmentation. Avoid simple counts. MUST-INCLUDE 
        Question Types: Time-Series/Trend Analysis, Cross-Table Correlation (requiring a JOIN), Distribution/Anomaly 
        Detection. Return ONLY the JSON array of strings. Example: {{"plan": ["Question 1?", "Question 2?"]}} 
        **Database Schema:** ``` {schema} ``` """
        response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.2)
        try:
            data = json.loads(response.choices[0].message.content)
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"[Orchestrator:create_initial_plan] Plan generated successfully.")
                    return value
            return ["Analyze the main table."]
        except Exception as e:
            print(f"[Orchestrator:create_initial_plan] ERROR: Failed to decode plan, using fallback: {e}")
            return ["Count rows in all tables.", "Describe the schema."]

    def process_evaluation(self, evaluation: dict, session_memory: list, analysis_plan: list):
        print("[Orchestrator:process_evaluation] Processing evaluation.")
        session_memory.append(evaluation['finding'])
        if evaluation.get('new_hypotheses'):
            print(f"[Orchestrator:process_evaluation] Adding {len(evaluation['new_hypotheses'])} new hypotheses to plan.")
            analysis_plan[:0] = evaluation['new_hypotheses']

class SQLCoder(BaseAgent):
    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.inspector = inspect(engine)
        self.table_names = [name for name in self.inspector.get_table_names() if name != 'alembic_version']
        self.db = SQLDatabase(engine=engine, include_tables=self.table_names, sample_rows_in_table_info=3)
        self.llm = ChatOpenAI(model=self.model, temperature=0, api_key=config.API_KEY)

    def run(self, question: str) -> dict:
        print(f"[SQLCoder:run] Starting for question: '{question[:50]}...'")
        table_info = self.db.get_table_info()
        escaped_table_info = table_info.replace("{", "{{").replace("}", "}}")
        system_prompt = f"""You are an expert PostgreSQL data analyst. Your task is to write a single, syntactically 
        correct SQL query to answer the user's question. **VERY IMPORTANT**: You have been provided with the complete 
        schema and sample rows for all tables below. Use this information directly. You MUST NOT use `sql_db_schema` 
        or `sql_db_list_tables`. Go straight to writing the query using `sql_db_query`. **DATABASE SCHEMA AND SAMPLE 
        ROWS:** ```sql {escaped_table_info} ``` Think step-by-step to construct the query. After thinking, 
        respond with ONLY the final SQL query in the correct tool format. """
        try:
            agent_executor = create_sql_agent(llm=self.llm, db=self.db, agent_type="openai-tools", verbose=True, agent_kwargs={"system_message": system_prompt}, handle_parsing_errors=True)
            result = agent_executor.invoke({"input": question})
            sql_query = "Could not extract SQL query."
            if 'intermediate_steps' in result and result['intermediate_steps']:
                for step in result['intermediate_steps']:
                    if isinstance(step, tuple) and len(step) > 0 and hasattr(step[0], 'tool_input'):
                        tool_input = step[0].tool_input
                        if isinstance(tool_input, dict) and 'query' in tool_input:
                            sql_query = tool_input['query']
                            break
            df = pd.DataFrame()
            if sql_query and "Could not extract" not in sql_query:
                print(f"[SQLCoder:run] RELIABLE EXECUTION: Running extracted SQL directly.\n--- Query ---\n{sql_query}\n-----------")
                try:
                    with self.engine.connect() as connection:
                        df = pd.read_sql(sql_query, connection)
                    print(f"[SQLCoder:run] SUCCESS: Retrieved {len(df)} rows from DB.")
                except Exception as e:
                    print(f"[SQLCoder:run] RELIABLE EXECUTION FAILED: {e}")
                    return {"question": question, "data": df, "raw_output": f"Query execution failed: {e}", "error": str(e)}

            return_value = {"question": question, "data": df, "raw_output": result.get('output', ''), "error": None}
            print(f"[SQLCoder:run] FINAL RETURN. DataFrame rows: {len(return_value['data'])}. Returning object to task loop.")
            return return_value

        except Exception as e:
            print(f"[SQLCoder:run] CRITICAL ERROR: {e}")
            return {"question": question, "data": pd.DataFrame(), "raw_output": f"An error occurred: {e}", "error": str(e)}

class Critic(BaseAgent):
    def evaluate(self, execution_result: dict):
        df = execution_result.get('data', pd.DataFrame())
        question = execution_result.get('question', 'N/A')
        print(f"[Critic:evaluate] RECEIVED. Question: '{question[:50]}...'. DataFrame is_empty: {df.empty}. Rows: {len(df)}.")

        if execution_result.get('error'):
            print(f"[Critic:evaluate] Found error in result: {execution_result['error']}")
            return {"is_success": False, "finding": {"question": question, "summary": f"Query failed: {execution_result['error']}"}}
        if df.empty:
            print("[Critic:evaluate] DataFrame is empty. Returning 'no data' finding.")
            return {"is_success": True, "finding": {"question": question, "summary": "Query executed but returned no "
                                                                                     "data.", "chart_url": None,
                                                    "data_preview": None}, "new_hypotheses": []}

        print("[Critic:evaluate] Data found. Generating summary and chart.")
        chart_url = create_visualization(df, question)
        df_markdown_escaped = df.head(10).to_markdown(index=False).replace("{", "{{").replace("}", "}}")
        prompt = f"""You are a Senior Data Analyst. Analyze the query result. **Original Question:** "{question}" **Data Result (first 10 rows):** ```{df_markdown_escaped}``` **Your Tasks (JSON format):** 1. `summary` (string): Write a concise, insightful summary. 2. `new_hypotheses` (array of strings): Generate 1-3 new follow-up questions. **OUTPUT MUST BE A VALID JSON OBJECT.**"""
        response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        try:
            eval_data = json.loads(response.choices[0].message.content)
            print(f"[Critic:evaluate] PRODUCED finding. Summary: '{eval_data.get('summary', 'N/A')[:50]}...'")
            return {"is_success": True, "finding": {"question": question, "summary": eval_data.get("summary"), "chart_url": chart_url, "data_preview": df.head().to_dict('records')}, "new_hypotheses": eval_data.get("new_hypotheses", [])}
        except Exception as e:
            print(f"[Critic:evaluate] ERROR: Failed to parse evaluation: {e}")
            return {"is_success": False, "finding": {"question": question, "summary": f"Failed to parse critic "
                                                                                      f"evaluation: {e}"}}

class Storyteller(BaseAgent):
    def narrate(self, session_memory: list):
        print(f"[Storyteller:narrate] RECEIVED {len(session_memory)} findings to synthesize.")
        findings_text = "\n\n".join([f"### {idx + 1}. On: '{finding.get('question', 'N/A')}'\n**Summary:** {finding.get('summary', 'No summary available.')}" for idx, finding in enumerate(session_memory)])
        findings_text_escaped = findings_text.replace("{", "{{").replace("}", "}}")
        prompt = f"""You are a Lead Data Analyst. Synthesize the findings into a report. **Findings:** --- {findings_text_escaped} --- **Your Output (JSON Object):** Generate JSON with two keys: `executive_summary` (string summary) and `detailed_findings` (array of objects with `question`, `summary`, `chart_url`, `data_preview`). """
        response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        try:
            final_json = json.loads(response.choices[0].message.content)
            print("[Storyteller:narrate] Report synthesized successfully.")
            return final_json
        except Exception as e:
            print(f"[Storyteller:narrate] ERROR: Failed to generate final report JSON: {e}")
            return {"executive_summary": "Failed to generate the final report.", "detailed_findings": session_memory}