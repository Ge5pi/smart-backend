import ast
import io
import json
import uuid
import logging
from typing import Optional, Dict, Any, List

import boto3
import openai
import pandas as pd
import plotly.express as px
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import inspect, text

import config

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Клиенты и утилиты
s3_client = boto3.client(
    's3',
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
    region_name=config.AWS_DEFAULT_REGION
)

openai_client = openai.OpenAI(api_key=config.API_KEY)

LLM_SQL_DEBUG_MODE = True


def create_visualization(df: pd.DataFrame, question: str) -> str | None:
    """Создает визуализацию данных и загружает в S3"""
    if df.empty or len(df.columns) < 2 or len(df) < 2:
        logger.info("Недостаточно данных для создания визуализации")
        return None

    try:
        x_col, y_col = df.columns[0], df.columns[1]
        x_is_date = pd.api.types.is_datetime64_any_dtype(df[x_col])
        y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])

        fig = None
        if x_is_date and y_is_numeric:
            fig = px.line(df, x=x_col, y=y_col, title=question, markers=True)
        elif y_is_numeric:
            fig = px.bar(
                df.sort_values(by=y_col, ascending=False).head(20),
                x=x_col, y=y_col, title=question
            )

        if fig:
            img_bytes = io.BytesIO()
            fig.write_image(img_bytes, format="png", scale=2)
            img_bytes.seek(0)

            file_name = f"charts/{uuid.uuid4()}.png"
            s3_client.upload_fileobj(
                img_bytes,
                config.S3_BUCKET_NAME,
                file_name,
                ExtraArgs={'ContentType': 'image/png', 'ACL': 'public-read'}
            )

            return f"https://{config.S3_BUCKET_NAME}.s3.{config.AWS_DEFAULT_REGION}.amazonaws.com/{file_name}"

        return None

    except Exception as e:
        logger.error(f"Ошибка создания визуализации для вопроса '{question}': {e}")
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
        """Получает подробную информацию о схеме БД"""
        inspector = inspect(self.engine)
        schema_info = []

        for table_name in inspector.get_table_names():
            if table_name == 'alembic_version':
                continue

            columns = inspector.get_columns(table_name)
            column_info = []

            for col in columns:
                col_type = str(col['type'])
                nullable = "" if col['nullable'] else " NOT NULL"
                column_info.append(f"{col['name']} ({col_type}){nullable}")

            # Получаем информацию о количестве записей
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    count = result.scalar()
                    schema_info.append(f"Table '{table_name}' ({count} rows) with columns: {column_info}")
            except Exception as e:
                logger.warning(f"Не удалось получить количество записей для {table_name}: {e}")
                schema_info.append(f"Table '{table_name}' with columns: {column_info}")

        return "\n".join(schema_info) or "No tables found."

    def validate_data_availability(self) -> Dict[str, int]:
        """Проверяет доступность данных в таблицах"""
        inspector = inspect(self.engine)
        table_counts = {}

        for table_name in inspector.get_table_names():
            if table_name == 'alembic_version':
                continue

            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    count = result.scalar()
                    table_counts[table_name] = count
                    logger.info(f"Таблица {table_name}: {count} записей")
            except Exception as e:
                logger.error(f"Ошибка проверки таблицы {table_name}: {e}")
                table_counts[table_name] = 0

        return table_counts

    def create_initial_plan(self):
        """Создает план анализа с учетом доступности данных"""
        logger.info("Создание плана анализа...")

        # Проверяем доступность данных
        table_counts = self.validate_data_availability()
        empty_tables = [table for table, count in table_counts.items() if count == 0]

        if empty_tables:
            logger.warning(f"Найдены пустые таблицы: {empty_tables}")

        # Фильтруем таблицы с данными
        available_tables = [table for table, count in table_counts.items() if count > 0]

        if not available_tables:
            logger.error("Все таблицы пусты! Возвращаем базовый план.")
            return [
                "Проверить структуру базы данных",
                "Подсчитать количество записей в каждой таблице",
                "Проанализировать схему данных"
            ]

        schema = self.get_schema_details()

        # Создаем план с учетом доступных данных
        prompt = f"""Вы - главный аналитик данных. Ваша цель - найти глубокие, неочевидные, практические 
        инсайты. На основе предоставленной схемы создайте стратегический план анализа в виде JSON массива 
        из 5-7 исследовательских вопросов.

        ВАЖНО: Фокусируйтесь только на таблицах с данными: {available_tables}

        Пустые таблицы (избегайте их): {empty_tables}

        Типы вопросов которые ДОЛЖНЫ быть включены:
        - Анализ временных рядов/трендов
        - Корреляция между таблицами (требующая JOIN)
        - Анализ распределения/аномалий
        - Сегментация пользователей

        Избегайте простых подсчетов. Сосредоточьтесь на корреляциях, трендах, аномалиях.

        Верните ТОЛЬКО JSON массив строк. Пример: {{"plan": ["Вопрос 1?", "Вопрос 2?"]}}

        **Схема базы данных:**
        ```
        {schema}
        ```
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            data = json.loads(response.choices[0].message.content)
            for key, value in data.items():
                if isinstance(value, list):
                    logger.info("План анализа создан успешно")
                    return value

            # Fallback план
            return [f"Проанализировать данные в таблице {available_tables[0]}"]

        except Exception as e:
            logger.error(f"Ошибка создания плана: {e}")
            return [
                f"Подсчитать записи в таблице {available_tables[0]}",
                "Описать схему данных"
            ]

    def process_evaluation(self, evaluation: dict, session_memory: list, analysis_plan: list):
        """Обрабатывает результаты оценки"""
        logger.info("Обработка результатов оценки")
        session_memory.append(evaluation['finding'])

        if evaluation.get('new_hypotheses'):
            logger.info(f"Добавление {len(evaluation['new_hypotheses'])} новых гипотез в план")
            analysis_plan[:0] = evaluation['new_hypotheses']


class SQLCoder(BaseAgent):
    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.inspector = inspect(engine)
        self.table_names = [
            name for name in self.inspector.get_table_names()
            if name != 'alembic_version'
        ]
        self.db = SQLDatabase(
            engine=engine,
            include_tables=self.table_names,
            sample_rows_in_table_info=3
        )
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0,
            api_key=config.API_KEY
        )

    def validate_query_before_execution(self, query: str) -> bool:
        """Проверяет SQL запрос перед выполнением"""
        try:
            # Проверяем базовый синтаксис
            query_lower = query.lower().strip()

            # Проверяем, что это SELECT запрос
            if not query_lower.startswith('select'):
                logger.warning("Запрос не является SELECT")
                return False

            # Проверяем наличие таблиц в запросе
            for table_name in self.table_names:
                if table_name in query_lower:
                    return True

            logger.warning("В запросе не найдены известные таблицы")
            return False

        except Exception as e:
            logger.error(f"Ошибка валидации запроса: {e}")
            return False

    def execute_query_directly(self, query: str) -> pd.DataFrame:
        """Выполняет SQL запрос напрямую"""
        try:
            with self.engine.connect() as connection:
                logger.info(f"Выполнение запроса: {query[:100]}...")
                df = pd.read_sql(query, connection)
                logger.info(f"Успешно получено {len(df)} строк")
                return df
        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            return pd.DataFrame()

    def run(self, question: str) -> dict:
        """Выполняет анализ вопроса и возвращает результат"""
        logger.info(f"Начало анализа вопроса: '{question[:50]}...'")

        # Получаем информацию о таблицах
        table_info = self.db.get_table_info()

        # Создаем улучшенный системный промпт
        system_prompt = f"""Вы - эксперт по PostgreSQL анализу данных. Ваша задача - написать единственный, 
        синтаксически корректный SQL запрос для ответа на вопрос пользователя.

        **КРИТИЧЕСКИ ВАЖНО**: 
        1. Используйте ТОЛЬКО таблицы из предоставленной схемы
        2. Всегда проверяйте, что таблицы содержат данные
        3. Начинайте с простых запросов, если не уверены в структуре данных
        4. Используйте LIMIT для ограничения результатов

        **СХЕМА БАЗЫ ДАННЫХ И ПРИМЕРЫ ДАННЫХ:**
        ```sql
        {table_info}
        ```

        **РЕКОМЕНДАЦИИ ПО НАПИСАНИЮ SQL:**
        - Избегайте декартовых соединений! Всегда используйте конкретные JOIN ключи
        - Для сравнений между таблицами используйте агрегации (MAX, MIN, COUNT, etc.)
        - Предпочитайте LEFT JOIN когда данные могут отсутствовать
        - Ограничивайте результаты с помощью LIMIT (максимум 100 строк)
        - Используйте WHERE для фильтрации данных
        - Группируйте по user_id или другим ключам, а не по email или status

        **ВАЖНО**: Если вопрос неясен, выберите консервативный аналитический подход.

        Подумайте пошагово и напишите ТОЛЬКО финальный SQL запрос.
        """

        try:
            # Пробуем использовать агента
            agent_executor = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_type="openai-tools",
                verbose=True,
                agent_kwargs={"system_message": system_prompt},
                handle_parsing_errors=True
            )

            result = agent_executor.invoke({"input": question})

            # Извлекаем SQL запрос
            sql_query = "Could not extract SQL query."
            if 'intermediate_steps' in result and result['intermediate_steps']:
                for step in result['intermediate_steps']:
                    if (isinstance(step, tuple) and len(step) > 0 and
                            hasattr(step[0], 'tool_input')):
                        tool_input = step[0].tool_input
                        if isinstance(tool_input, dict) and 'query' in tool_input:
                            sql_query = tool_input['query']
                            break

            df = pd.DataFrame()

            # Выполняем запрос напрямую, если удалось его извлечь
            if sql_query and "Could not extract" not in sql_query:
                if self.validate_query_before_execution(sql_query):
                    df = self.execute_query_directly(sql_query)
                else:
                    logger.warning("Запрос не прошел валидацию")

            # Если результат пустой, пробуем более простой подход
            if df.empty:
                logger.warning("Получен пустой результат, пробуем упрощенный запрос")

                # Создаем простой запрос для проверки данных
                simple_queries = [
                    f"SELECT COUNT(*) as count FROM {table} LIMIT 1"
                    for table in self.table_names[:3]
                ]

                for simple_query in simple_queries:
                    test_df = self.execute_query_directly(simple_query)
                    if not test_df.empty:
                        logger.info(f"Найдены данные в таблице: {simple_query}")
                        break

            # Логируем результат
            if df.empty:
                logger.warning("SQL вернул пустой результат")
                with open("empty_queries.log", "a") as f:
                    f.write(f"\n[EMPTY RESULT] {question}\nSQL: {sql_query}\n---\n")

            return {
                "question": question,
                "data": df,
                "raw_output": result.get('output', ''),
                "sql_query": sql_query,
                "error": None
            }

        except Exception as e:
            logger.error(f"Критическая ошибка в SQLCoder: {e}")
            return {
                "question": question,
                "data": pd.DataFrame(),
                "raw_output": f"Произошла ошибка: {e}",
                "sql_query": None,
                "error": str(e)
            }


class Critic(BaseAgent):
    def evaluate(self, execution_result: dict):
        """Оценивает результат выполнения"""
        df = execution_result.get('data', pd.DataFrame())
        question = execution_result.get('question', 'N/A')

        logger.info(f"Оценка результата. Вопрос: '{question[:50]}...'. "
                    f"Пустой DataFrame: {df.empty}. Строк: {len(df)}")

        # Обрабатываем ошибки
        if execution_result.get('error'):
            logger.error(f"Найдена ошибка в результате: {execution_result['error']}")
            return {
                "is_success": False,
                "finding": {
                    "question": question,
                    "summary": f"Запрос завершился с ошибкой: {execution_result['error']}"
                }
            }

        # Обрабатываем пустые результаты
        if df.empty:
            logger.warning("DataFrame пустой")
            return {
                "is_success": True,
                "finding": {
                    "question": question,
                    "summary": "Запрос выполнен, но данные не найдены. Возможно, таблицы пусты или условия фильтрации слишком строгие.",
                    "chart_url": None,
                    "data_preview": None
                },
                "new_hypotheses": [
                    "Проверить наличие данных в базовых таблицах",
                    "Упростить условия фильтрации в запросах"
                ]
            }

        # Обрабатываем успешные результаты
        logger.info("Данные найдены, создаем сводку и график")

        chart_url = create_visualization(df, question)
        df_markdown = df.head(10).to_markdown(index=False)

        prompt = f"""Вы - старший аналитик данных. Проанализируйте результат запроса.

        **Исходный вопрос:** "{question}"

        **Результат данных (первые 10 строк):**
        ```
        {df_markdown}
        ```

        **Ваши задачи (JSON формат):**
        1. `summary` (строка): Напишите краткую, информативную сводку
        2. `new_hypotheses` (массив строк): Сгенерируйте 1-3 новых уточняющих вопроса

        **ВЫВОД ДОЛЖЕН БЫТЬ ВАЛИДНЫМ JSON ОБЪЕКТОМ.**
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            eval_data = json.loads(response.choices[0].message.content)

            logger.info(f"Создана сводка: '{eval_data.get('summary', 'N/A')[:50]}...'")

            return {
                "is_success": True,
                "finding": {
                    "question": question,
                    "summary": eval_data.get("summary"),
                    "chart_url": chart_url,
                    "data_preview": df.head().to_dict('records')
                },
                "new_hypotheses": eval_data.get("new_hypotheses", [])
            }

        except Exception as e:
            logger.error(f"Ошибка парсинга оценки: {e}")
            return {
                "is_success": False,
                "finding": {
                    "question": question,
                    "summary": f"Не удалось обработать результат оценки: {e}"
                }
            }


class Storyteller(BaseAgent):
    def narrate(self, session_memory: list):
        """Создает итоговый отчет на основе всех результатов"""
        logger.info(f"Создание отчета на основе {len(session_memory)} результатов")

        if not session_memory:
            return {
                "executive_summary": "Анализ не дал результатов. Возможно, база данных пуста или требует дополнительной настройки.",
                "detailed_findings": []
            }

        # Фильтруем только успешные результаты
        successful_findings = [
            finding for finding in session_memory
            if finding.get('summary') and 'ошибка' not in finding.get('summary', '').lower()
        ]

        if not successful_findings:
            return {
                "executive_summary": "Анализ выполнен, но все запросы вернули пустые результаты. Рекомендуется проверить наличие данных в базе и корректность запросов.",
                "detailed_findings": session_memory
            }

        findings_text = "\n\n".join([
            f"### {idx + 1}. По вопросу: '{finding.get('question', 'N/A')}'\n"
            f"Результат: {finding.get('summary', 'Нет описания')}"
            for idx, finding in enumerate(successful_findings)
        ])

        prompt = f"""Вы - ведущий аналитик данных. Синтезируйте результаты в отчет.

        Результаты анализа:
        ---
        {findings_text}
        ---

        Создайте JSON объект с двумя ключами:
        - `executive_summary` (строка): Краткое резюме основных инсайтов
        - `detailed_findings` (массив объектов): Детальные результаты с полями question, summary, chart_url, data_preview

        **ВАЖНО**: Сфокусируйтесь на практических инсайтах и рекомендациях.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            final_json = json.loads(response.choices[0].message.content)
            logger.info("Итоговый отчет создан успешно")

            return final_json

        except Exception as e:
            logger.error(f"Ошибка создания итогового отчета: {e}")
            return {
                "executive_summary": "Не удалось создать итоговый отчет из-за технической ошибки.",
                "detailed_findings": session_memory
            }
