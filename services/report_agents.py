import ast
import io
import json
import uuid
import logging
import re
from typing import Optional, Dict, Any, List, Tuple

import boto3
import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

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
    """Создает умную визуализацию данных и загружает в S3"""
    if df.empty:
        logger.info("DataFrame пустой, визуализация невозможна")
        return None

    try:
        # Если одна колонка - создаем распределение
        if len(df.columns) == 1:
            col = df.columns[0]
            if pd.api.types.is_numeric_dtype(df[col]):
                fig = px.histogram(df, x=col, title=f"Распределение: {question}")
            else:
                value_counts = df[col].value_counts().head(20)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                             title=f"Топ значения: {question}")

        # Если две колонки - создаем соответствующую визуализацию
        elif len(df.columns) == 2:
            x_col, y_col = df.columns[0], df.columns[1]
            x_is_date = pd.api.types.is_datetime64_any_dtype(df[x_col])
            y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])

            if x_is_date and y_is_numeric:
                fig = px.line(df, x=x_col, y=y_col, title=question, markers=True)
            elif y_is_numeric:
                # Сортируем и берем топ для лучшей визуализации
                df_sorted = df.sort_values(by=y_col, ascending=False).head(20)
                fig = px.bar(df_sorted, x=x_col, y=y_col, title=question)
            else:
                # Категориальные данные
                fig = px.bar(df.head(20), x=x_col, y=y_col, title=question)

        # Если больше двух колонок - создаем корреляционную матрицу или scatter
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                # Корреляционная матрица для числовых данных
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title=f"Корреляция: {question}",
                                color_continuous_scale='RdBu_r')
            else:
                # Группировка по первой колонке
                first_col = df.columns[0]
                grouped = df.groupby(first_col).size().reset_index(name='count')
                fig = px.bar(grouped.head(20), x=first_col, y='count',
                             title=f"Группировка: {question}")

        if fig:
            # Улучшаем внешний вид графика
            fig.update_layout(
                title_font_size=16,
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                showlegend=True,
                height=500,
                margin=dict(l=50, r=50, t=80, b=50)
            )

            img_bytes = io.BytesIO()
            fig.write_image(img_bytes, format="png", scale=2, width=800, height=500)
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
        logger.error(f"Ошибка создания визуализации: {e}")
        return None


class BaseAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.client = openai_client
        self.model = model


class Orchestrator(BaseAgent):
    def __init__(self, engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine
        self.schema_cache = None
        self.table_stats_cache = None

    def get_comprehensive_schema(self) -> Dict[str, Any]:
        """Получает полную информацию о схеме с примерами данных"""
        if self.schema_cache:
            return self.schema_cache

        inspector = inspect(self.engine)
        schema_info = {}

        for table_name in inspector.get_table_names():
            if table_name == 'alembic_version':
                continue

            try:
                columns = inspector.get_columns(table_name)

                # Получаем статистику по таблице
                with self.engine.connect() as conn:
                    # Количество записей
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.scalar()

                    if row_count > 0:
                        # Примеры данных
                        sample_result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 5"))
                        sample_data = sample_result.fetchall()

                        # Статистика по колонкам
                        column_stats = {}
                        for col in columns:
                            col_name = col['name']
                            col_type = str(col['type'])

                            # Пытаемся получить уникальные значения для категориальных полей
                            if 'varchar' in col_type.lower() or 'text' in col_type.lower():
                                unique_result = conn.execute(text(
                                    f"SELECT COUNT(DISTINCT {col_name}) as unique_count FROM {table_name}"
                                ))
                                unique_count = unique_result.scalar()

                                if unique_count <= 50:  # Показываем примеры для категориальных
                                    examples_result = conn.execute(text(
                                        f"SELECT DISTINCT {col_name} FROM {table_name} LIMIT 10"
                                    ))
                                    examples = [row[0] for row in examples_result.fetchall()]
                                    column_stats[col_name] = {
                                        'type': col_type,
                                        'unique_count': unique_count,
                                        'examples': examples
                                    }
                                else:
                                    column_stats[col_name] = {
                                        'type': col_type,
                                        'unique_count': unique_count
                                    }
                            else:
                                column_stats[col_name] = {'type': col_type}

                        schema_info[table_name] = {
                            'row_count': row_count,
                            'columns': column_stats,
                            'sample_data': sample_data
                        }
                    else:
                        schema_info[table_name] = {'row_count': 0, 'columns': {}, 'sample_data': []}

            except Exception as e:
                logger.error(f"Ошибка анализа таблицы {table_name}: {e}")
                schema_info[table_name] = {'row_count': 0, 'columns': {}, 'sample_data': []}

        self.schema_cache = schema_info
        return schema_info

    def create_intelligent_plan(self) -> List[str]:
        """Создает умный план анализа на основе реальной структуры данных"""
        logger.info("Создание интеллектуального плана анализа...")

        schema = self.get_comprehensive_schema()

        # Фильтруем только таблицы с данными
        populated_tables = {k: v for k, v in schema.items() if v['row_count'] > 0}

        if not populated_tables:
            logger.error("Нет таблиц с данными!")
            return ["Проверить наличие данных в базе"]

        # Анализируем структуру для создания умного плана
        plan = []

        # 1. Базовая статистика по каждой таблице
        for table_name, info in populated_tables.items():
            plan.append(f"Проанализировать общую статистику таблицы {table_name}")

        # 2. Поиск временных колонок для трендового анализа
        for table_name, info in populated_tables.items():
            for col_name, col_info in info['columns'].items():
                if any(keyword in col_name.lower() for keyword in ['date', 'time', 'created', 'updated']):
                    plan.append(f"Проанализировать временные тренды в {table_name} по колонке {col_name}")

        # 3. Анализ категориальных данных
        for table_name, info in populated_tables.items():
            for col_name, col_info in info['columns'].items():
                if col_info.get('unique_count', 0) > 1 and col_info.get('unique_count', 0) <= 50:
                    plan.append(f"Проанализировать распределение {col_name} в таблице {table_name}")

        # 4. Поиск связей между таблицами
        table_names = list(populated_tables.keys())
        if len(table_names) > 1:
            for i, table1 in enumerate(table_names):
                for table2 in table_names[i + 1:]:
                    # Ищем потенциальные внешние ключи
                    cols1 = set(populated_tables[table1]['columns'].keys())
                    cols2 = set(populated_tables[table2]['columns'].keys())

                    # Общие колонки могут быть связями
                    common_cols = cols1.intersection(cols2)
                    if common_cols:
                        common_col = list(common_cols)[0]
                        plan.append(f"Найти корреляции между {table1} и {table2} через {common_col}")

        # 5. Поиск аномалий в числовых данных
        for table_name, info in populated_tables.items():
            numeric_cols = [col for col, col_info in info['columns'].items()
                            if any(t in col_info['type'].lower() for t in ['int', 'float', 'numeric', 'decimal'])]
            if numeric_cols:
                plan.append(f"Найти аномалии в числовых данных таблицы {table_name}")

        # Ограничиваем план разумным количеством вопросов
        return plan[:8]

    def process_evaluation(self, evaluation: dict, session_memory: list, analysis_plan: list):
        """Обрабатывает результаты оценки и добавляет новые гипотезы"""
        logger.info("Обработка результатов оценки")

        if evaluation.get('finding'):
            session_memory.append(evaluation['finding'])

        if evaluation.get('new_hypotheses'):
            # Добавляем новые гипотезы в начало плана
            new_hypotheses = evaluation['new_hypotheses'][:3]  # Ограничиваем количество
            analysis_plan[:0] = new_hypotheses
            logger.info(f"Добавлено {len(new_hypotheses)} новых гипотез")


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
            sample_rows_in_table_info=5
        )
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0.1,
            api_key=config.API_KEY
        )

    def get_fallback_queries(self, question: str) -> List[str]:
        """Генерирует fallback запросы для получения хотя бы каких-то данных"""
        fallback_queries = []

        # Простые запросы для каждой таблицы
        for table in self.table_names:
            fallback_queries.extend([
                f"SELECT * FROM {table} LIMIT 10",
                f"SELECT COUNT(*) as total_rows FROM {table}",
                f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT 5"
            ])

        # Если в вопросе есть ключевые слова, создаем специфичные запросы
        question_lower = question.lower()

        for table in self.table_names:
            try:
                columns = self.inspector.get_columns(table)

                # Поиск колонок по ключевым словам
                for col in columns:
                    col_name = col['name'].lower()

                    # Временные колонки
                    if any(keyword in col_name for keyword in ['date', 'time', 'created', 'updated']):
                        fallback_queries.append(
                            f"SELECT DATE({col['name']}) as date, COUNT(*) as count FROM {table} "
                            f"GROUP BY DATE({col['name']}) ORDER BY date DESC LIMIT 20"
                        )

                    # Категориальные колонки
                    if 'varchar' in str(col['type']).lower() or 'text' in str(col['type']).lower():
                        fallback_queries.append(
                            f"SELECT {col['name']}, COUNT(*) as count FROM {table} "
                            f"GROUP BY {col['name']} ORDER BY count DESC LIMIT 20"
                        )

                    # Числовые колонки
                    if any(t in str(col['type']).lower() for t in ['int', 'float', 'numeric', 'decimal']):
                        fallback_queries.append(
                            f"SELECT AVG({col['name']}) as avg_value, MIN({col['name']}) as min_value, "
                            f"MAX({col['name']}) as max_value FROM {table}"
                        )

            except Exception as e:
                logger.error(f"Ошибка генерации fallback запросов для {table}: {e}")

        return fallback_queries

    def execute_query_safely(self, query: str) -> pd.DataFrame:
        """Безопасно выполняет SQL запрос с обработкой ошибок"""
        try:
            # Добавляем LIMIT если его нет
            if 'limit' not in query.lower() and 'count(' not in query.lower():
                query = f"{query.rstrip(';')} LIMIT 100"

            with self.engine.connect() as connection:
                logger.info(f"Выполнение запроса: {query[:100]}...")
                df = pd.read_sql(query, connection)
                logger.info(f"Получено {len(df)} строк, {len(df.columns)} колонок")
                return df

        except Exception as e:
            logger.error(f"Ошибка выполнения запроса: {e}")
            return pd.DataFrame()

    def extract_sql_from_agent_response(self, agent_result: dict) -> str:
        """Извлекает SQL запрос из ответа агента"""
        try:
            # Проверяем intermediate_steps
            if 'intermediate_steps' in agent_result:
                for step in agent_result['intermediate_steps']:
                    if isinstance(step, tuple) and len(step) >= 2:
                        action, observation = step
                        if hasattr(action, 'tool_input'):
                            tool_input = action.tool_input
                            if isinstance(tool_input, dict) and 'query' in tool_input:
                                return tool_input['query']

            # Пытаемся найти SQL в тексте ответа
            output_text = agent_result.get('output', '')
            sql_patterns = [
                r'```sql\n(.*?)\n```',
                r'```\n(SELECT.*?)\n```',
                r'Query:\s*(SELECT.*?)(?:\n|$)',
                r'(SELECT.*?)(?:\n|$)'
            ]

            for pattern in sql_patterns:
                match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(1).strip()

            return None

        except Exception as e:
            logger.error(f"Ошибка извлечения SQL: {e}")
            return None

    def run(self, question: str) -> dict:
        """Выполняет анализ с гарантированным получением результата"""
        logger.info(f"Анализ вопроса: '{question}'")

        # Получаем информацию о таблицах
        table_info = self.db.get_table_info()

        # Улучшенный системный промпт
        system_prompt = f"""Вы - эксперт по анализу данных PostgreSQL. 
        Ваша задача - написать SQL запрос, который ОБЯЗАТЕЛЬНО вернет данные.

        **КРИТИЧЕСКИ ВАЖНО:**
        1. ВСЕГДА используйте LIMIT для ограничения результатов
        2. Начинайте с простых запросов, постепенно усложняя
        3. Если не уверены в структуре - делайте SELECT * FROM table LIMIT 10
        4. Используйте агрегации (COUNT, AVG, SUM) для получения статистики
        5. Группируйте данные для получения интересных инсайтов

        **СХЕМА БАЗЫ ДАННЫХ:**
        {table_info}

        **СТРАТЕГИЯ НАПИСАНИЯ SQL:**
        - Для временных трендов: GROUP BY DATE/DATE_TRUNC
        - Для категориальных данных: GROUP BY с COUNT(*)
        - Для числовых данных: AVG, MIN, MAX, PERCENTILE
        - Для корреляций: JOIN таблицы по общим ключам

        Напишите SQL запрос, который гарантированно вернет данные.
        """

        df = pd.DataFrame()
        sql_query = None
        error_msg = None

        try:
            # Пытаемся использовать агента
            agent_executor = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_type="openai-tools",
                verbose=True,
                agent_kwargs={"system_message": system_prompt},
                handle_parsing_errors=True,
                max_iterations=3
            )

            agent_result = agent_executor.invoke({"input": question})
            sql_query = self.extract_sql_from_agent_response(agent_result)

            # Выполняем извлеченный запрос
            if sql_query:
                df = self.execute_query_safely(sql_query)

        except Exception as e:
            logger.error(f"Ошибка агента: {e}")
            error_msg = str(e)

        # Если агент не дал результата, используем fallback стратегию
        if df.empty:
            logger.warning("Агент не вернул данные, используем fallback стратегию")

            fallback_queries = self.get_fallback_queries(question)

            for fallback_query in fallback_queries:
                try:
                    df = self.execute_query_safely(fallback_query)
                    if not df.empty:
                        sql_query = fallback_query
                        logger.info(f"Fallback запрос успешен: {fallback_query[:50]}...")
                        break
                except Exception as e:
                    logger.error(f"Ошибка fallback запроса: {e}")
                    continue

        # Если все еще пусто, делаем базовый запрос
        if df.empty and self.table_names:
            logger.warning("Все запросы неуспешны, выполняем базовый запрос")

            for table in self.table_names:
                try:
                    basic_query = f"SELECT * FROM {table} LIMIT 10"
                    df = self.execute_query_safely(basic_query)
                    if not df.empty:
                        sql_query = basic_query
                        logger.info(f"Базовый запрос успешен для таблицы {table}")
                        break
                except Exception as e:
                    logger.error(f"Ошибка базового запроса для {table}: {e}")
                    continue

        # Формируем результат
        result = {
            "question": question,
            "data": df,
            "sql_query": sql_query,
            "error": error_msg,
            "row_count": len(df),
            "column_count": len(df.columns) if not df.empty else 0
        }

        logger.info(f"Результат: {len(df)} строк, {len(df.columns) if not df.empty else 0} колонок")
        return result


class Critic(BaseAgent):
    def evaluate(self, execution_result: dict) -> dict:
        """Умная оценка результатов с генерацией инсайтов"""
        df = execution_result.get('data', pd.DataFrame())
        question = execution_result.get('question', 'N/A')
        sql_query = execution_result.get('sql_query', 'N/A')

        logger.info(f"Оценка результата для вопроса: '{question[:50]}...'")
        logger.info(f"Данные: {len(df)} строк, {len(df.columns) if not df.empty else 0} колонок")

        # Обработка ошибок
        if execution_result.get('error'):
            return {
                "is_success": False,
                "finding": {
                    "question": question,
                    "summary": f"Техническая ошибка: {execution_result['error']}",
                    "sql_query": sql_query
                },
                "new_hypotheses": []
            }

        # Обработка пустых результатов
        if df.empty:
            return {
                "is_success": True,
                "finding": {
                    "question": question,
                    "summary": "Запрос выполнен успешно, но не вернул данные. Возможно, условия фильтрации слишком строгие или данные отсутствуют.",
                    "sql_query": sql_query,
                    "chart_url": None,
                    "data_preview": None
                },
                "new_hypotheses": [
                    "Проверить базовую статистику по всем таблицам",
                    "Упростить условия поиска данных"
                ]
            }

        # Успешный результат - создаем детальный анализ
        try:
            # Создаем визуализацию
            chart_url = create_visualization(df, question)

            # Анализируем данные
            data_analysis = self._analyze_dataframe(df)

            # Создаем превью данных
            data_preview = df.head(10).to_dict('records')

            # Генерируем инсайты через GPT
            insights = self._generate_insights(df, question, data_analysis)

            # Генерируем новые гипотезы
            new_hypotheses = self._generate_hypotheses(df, question, data_analysis)

            return {
                "is_success": True,
                "finding": {
                    "question": question,
                    "summary": insights,
                    "sql_query": sql_query,
                    "chart_url": chart_url,
                    "data_preview": data_preview,
                    "data_stats": data_analysis
                },
                "new_hypotheses": new_hypotheses
            }

        except Exception as e:
            logger.error(f"Ошибка оценки результата: {e}")
            return {
                "is_success": False,
                "finding": {
                    "question": question,
                    "summary": f"Ошибка анализа результата: {e}",
                    "sql_query": sql_query
                }
            }

    def _analyze_dataframe(self, df: pd.DataFrame) -> dict:
        """Анализирует DataFrame и возвращает статистику"""
        analysis = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_stats": {},
            "categorical_stats": {}
        }

        # Анализ числовых колонок
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            analysis["numeric_stats"][col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }

        # Анализ категориальных колонок
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            analysis["categorical_stats"][col] = {
                "unique_count": len(value_counts),
                "top_values": value_counts.head(5).to_dict()
            }

        return analysis

    def _generate_insights(self, df: pd.DataFrame, question: str, data_analysis: dict) -> str:
        """Генерирует инсайты на основе данных"""
        try:
            # Подготавливаем данные для анализа
            df_sample = df.head(20).to_string()

            prompt = f"""Проанализируйте данные и создайте краткий, содержательный инсайт.

            Исходный вопрос: "{question}"

            Статистика данных:
            - Размер: {data_analysis['shape']}
            - Колонки: {data_analysis['columns']}
            - Числовая статистика: {data_analysis['numeric_stats']}
            - Категориальная статистика: {data_analysis['categorical_stats']}

            Примеры данных:
            {df_sample}

            Напишите краткий (2-3 предложения) инсайт, который:
            1. Отвечает на исходный вопрос
            2. Выделяет ключевые паттерны или тренды
            3. Предлагает практические выводы

            Ответ должен быть на русском языке и содержать конкретные цифры из данных.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Ошибка генерации инсайтов: {e}")
            return f"Найдено {len(df)} записей по запросу '{question}'. Данные включают {len(df.columns)} колонок."

    def _generate_hypotheses(self, df: pd.DataFrame, question: str, data_analysis: dict) -> List[str]:
        """Генерирует новые гипотезы для исследования"""
        hypotheses = []

        # На основе числовых данных
        if data_analysis['numeric_stats']:
            numeric_cols = list(data_analysis['numeric_stats'].keys())
            if len(numeric_cols) > 1:
                hypotheses.append(f"Проанализировать корреляцию между {numeric_cols[0]} и {numeric_cols[1]}")

            for col in numeric_cols:
                stats = data_analysis['numeric_stats'][col]
                if stats['std'] > stats['mean']:  # Высокая вариативность
                    hypotheses.append(f"Найти аномалии в колонке {col}")

        # На основе категориальных данных
        if data_analysis['categorical_stats']:
            for col, stats in data_analysis['categorical_stats'].items():
                if stats['unique_count'] > 1:
                    hypotheses.append(f"Проанализировать распределение по группам в колонке {col}")

        # На основе временных паттернов
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            hypotheses.append(f"Проанализировать временные тренды по колонке {date_cols[0]}")

        # Ограничиваем количество гипотез
        return hypotheses[:3]


class Storyteller(BaseAgent):
    def narrate(self, session_memory: list) -> dict:
        """Создает комплексный отчет с глубоким анализом"""
        logger.info(f"Создание итогового отчета на основе {len(session_memory)} результатов")

        if not session_memory:
            return {
                "executive_summary": "Анализ не дал результатов. База данных может быть пустой или недоступной.",
                "detailed_findings": [],
                "recommendations": ["Проверить подключение к базе данных", "Убедиться в наличии данных"],
                "key_metrics": {}
            }

        # Фильтруем только успешные результаты
        successful_findings = [
            finding for finding in session_memory
            if finding.get('summary') and 'ошибка' not in finding.get('summary', '').lower()
        ]

        if not successful_findings:
            return {
                "executive_summary": "Анализ выполнен, но большинство запросов не вернули данных. Рекомендуется проверить структуру данных.",
                "detailed_findings": session_memory,
                "recommendations": [
                    "Проверить наличие данных в основных таблицах",
                    "Упростить критерии поиска",
                    "Проверить права доступа к данным"
                ],
                "key_metrics": {"total_queries": len(session_memory), "successful_queries": 0}
            }

        # Анализируем успешные результаты
        try:
            # Собираем ключевые метрики
            key_metrics = self._extract_key_metrics(successful_findings)

            # Генерируем executive summary
            executive_summary = self._generate_executive_summary(successful_findings, key_metrics)

            # Создаем рекомендации
            recommendations = self._generate_recommendations(successful_findings)

            # Обогащаем детальные результаты
            detailed_findings = self._enrich_findings(successful_findings)

            return {
                "executive_summary": executive_summary,
                "detailed_findings": detailed_findings,
                "recommendations": recommendations,
                "key_metrics": key_metrics,
                "analysis_quality": {
                    "total_queries": len(session_memory),
                    "successful_queries": len(successful_findings),
                    "success_rate": len(successful_findings) / len(session_memory) * 100
                }
            }

        except Exception as e:
            logger.error(f"Ошибка создания итогового отчета: {e}")
            return {
                "executive_summary": f"Не удалось создать полный отчет из-за ошибки: {e}",
                "detailed_findings": session_memory,
                "recommendations": ["Проверить техническую конфигурацию системы"],
                "key_metrics": {}
            }

    def _extract_key_metrics(self, findings: List[dict]) -> dict:
        """Извлекает ключевые метрики из результатов"""
        metrics = {
            "total_findings": len(findings),
            "queries_with_charts": 0,
            "queries_with_data": 0,
            "average_rows_per_query": 0,
            "data_types_analyzed": set(),
            "tables_analyzed": set()
        }

        total_rows = 0
        for finding in findings:
            if finding.get('chart_url'):
                metrics["queries_with_charts"] += 1

            if finding.get('data_preview'):
                metrics["queries_with_data"] += 1
                total_rows += len(finding['data_preview'])

            # Анализируем SQL запросы для извлечения таблиц
            sql_query = finding.get('sql_query', '')
            if sql_query:
                # Простой парсинг для извлечения таблиц
                import re
                tables = re.findall(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
                metrics["tables_analyzed"].update(tables)

                joins = re.findall(r'JOIN\s+(\w+)', sql_query, re.IGNORECASE)
                metrics["tables_analyzed"].update(joins)

        if metrics["queries_with_data"] > 0:
            metrics["average_rows_per_query"] = total_rows / metrics["queries_with_data"]

        # Преобразуем sets в списки для JSON сериализации
        metrics["data_types_analyzed"] = list(metrics["data_types_analyzed"])
        metrics["tables_analyzed"] = list(metrics["tables_analyzed"])

        return metrics

    def _generate_executive_summary(self, findings: List[dict], metrics: dict) -> str:
        """Генерирует executive summary с использованием GPT"""
        try:
            # Подготавливаем данные для анализа
            findings_text = "\n".join([
                f"- {finding.get('question', 'N/A')}: {finding.get('summary', 'N/A')}"
                for finding in findings[:10]  # Берем первые 10 для экономии токенов
            ])

            prompt = f"""Создайте executive summary для отчета по анализу данных.

            Ключевые метрики:
            - Всего успешных запросов: {metrics['total_findings']}
            - Запросов с графиками: {metrics['queries_with_charts']}
            - Среднее количество строк на запрос: {metrics['average_rows_per_query']:.1f}
            - Проанализированные таблицы: {', '.join(metrics['tables_analyzed'])}

            Основные результаты:
            {findings_text}

            Напишите краткое резюме (3-4 предложения), которое:
            1. Описывает общий объем проведенного анализа
            2. Выделяет главные инсайты и паттерны
            3. Указывает на практическую ценность результатов
            4. Делает выводы о состоянии данных

            Ответ на русском языке, профессиональным тоном.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Ошибка генерации executive summary: {e}")
            return f"Проведен анализ {len(findings)} запросов к базе данных. Получены данные из {len(metrics['tables_analyzed'])} таблиц. Создано {metrics['queries_with_charts']} визуализаций."

    def _generate_recommendations(self, findings: List[dict]) -> List[str]:
        """Генерирует рекомендации на основе результатов"""
        recommendations = []

        # Анализируем типы запросов для рекомендаций
        has_time_analysis = any(
            'врем' in finding.get('summary', '').lower() or 'дата' in finding.get('summary', '').lower() for finding in
            findings)
        has_correlations = any(
            'корреляц' in finding.get('summary', '').lower() or 'связ' in finding.get('summary', '').lower() for finding
            in findings)
        has_anomalies = any(
            'аномал' in finding.get('summary', '').lower() or 'выброс' in finding.get('summary', '').lower() for finding
            in findings)

        if has_time_analysis:
            recommendations.append("Продолжить мониторинг временных трендов для выявления сезонности")

        if has_correlations:
            recommendations.append("Углубить анализ корреляций для построения предиктивных моделей")

        if has_anomalies:
            recommendations.append("Внедрить систему автоматического обнаружения аномалий")

        # Общие рекомендации
        recommendations.extend([
            "Автоматизировать регулярное обновление ключевых метрик",
            "Создать дашборд для мониторинга основных показателей",
            "Провести более глубокий анализ выявленных паттернов"
        ])

        return recommendations[:5]  # Ограничиваем количество

    def _enrich_findings(self, findings: List[dict]) -> List[dict]:
        """Обогащает результаты дополнительной информацией"""
        enriched = []

        for i, finding in enumerate(findings):
            enriched_finding = finding.copy()
            enriched_finding['order'] = i + 1
            enriched_finding['category'] = self._categorize_finding(finding)
            enriched_finding['confidence'] = self._assess_confidence(finding)
            enriched.append(enriched_finding)

        return enriched

    def _categorize_finding(self, finding: dict) -> str:
        """Категоризирует результат анализа"""
        summary = finding.get('summary', '').lower()
        question = finding.get('question', '').lower()

        if any(keyword in summary or keyword in question for keyword in ['время', 'дата', 'тренд']):
            return "Временной анализ"
        elif any(keyword in summary or keyword in question for keyword in ['корреляц', 'связ', 'зависимост']):
            return "Корреляционный анализ"
        elif any(keyword in summary or keyword in question for keyword in ['распределен', 'группир', 'категор']):
            return "Категориальный анализ"
        elif any(keyword in summary or keyword in question for keyword in ['аномал', 'выброс', 'отклонен']):
            return "Анализ аномалий"
        elif any(keyword in summary or keyword in question for keyword in ['статистик', 'средн', 'сумм']):
            return "Статистический анализ"
        else:
            return "Общий анализ"

    def _assess_confidence(self, finding: dict) -> str:
        """Оценивает уверенность в результате"""
        data_preview = finding.get('data_preview', [])
        chart_url = finding.get('chart_url')

        if len(data_preview) > 50 and chart_url:
            return "Высокая"
        elif len(data_preview) > 10:
            return "Средняя"
        else:
            return "Низкая"


# Утилитарные функции для улучшения работы

def validate_database_connection(engine) -> bool:
    """Проверяет соединение с базой данных"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Ошибка подключения к БД: {e}")
        return False


def get_database_health_check(engine) -> dict:
    """Проверяет состояние базы данных"""
    health_check = {
        "connection": False,
        "tables": [],
        "total_rows": 0,
        "has_data": False
    }

    try:
        if validate_database_connection(engine):
            health_check["connection"] = True

            inspector = inspect(engine)
            tables = [t for t in inspector.get_table_names() if t != 'alembic_version']

            with engine.connect() as conn:
                for table in tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        health_check["tables"].append({"name": table, "rows": count})
                        health_check["total_rows"] += count
                    except Exception as e:
                        logger.error(f"Ошибка проверки таблицы {table}: {e}")
                        health_check["tables"].append({"name": table, "rows": 0})

            health_check["has_data"] = health_check["total_rows"] > 0

    except Exception as e:
        logger.error(f"Ошибка проверки здоровья БД: {e}")

    return health_check


# Основная функция для запуска полного анализа
def run_comprehensive_analysis(engine, max_questions: int = 10) -> dict:
    """Запускает полный анализ базы данных"""
    logger.info("Начало комплексного анализа базы данных")

    # Проверяем здоровье базы данных
    health_check = get_database_health_check(engine)
    if not health_check["connection"]:
        return {
            "error": "Не удалось подключиться к базе данных",
            "health_check": health_check
        }

    if not health_check["has_data"]:
        return {
            "error": "База данных не содержит данных",
            "health_check": health_check
        }

    # Инициализируем агентов
    orchestrator = Orchestrator(engine)
    sql_coder = SQLCoder(engine)
    critic = Critic()
    storyteller = Storyteller()

    # Создаем план анализа
    analysis_plan = orchestrator.create_intelligent_plan()
    session_memory = []

    logger.info(f"Создан план из {len(analysis_plan)} вопросов")

    # Выполняем анализ
    questions_processed = 0
    for question in analysis_plan:
        if questions_processed >= max_questions:
            break

        logger.info(f"Обрабатывается вопрос {questions_processed + 1}/{max_questions}: {question}")

        # Получаем данные
        execution_result = sql_coder.run(question)

        # Оцениваем результат
        evaluation = critic.evaluate(execution_result)

        # Обрабатываем результат
        orchestrator.process_evaluation(evaluation, session_memory, analysis_plan)

        questions_processed += 1

    # Создаем итоговый отчет
    final_report = storyteller.narrate(session_memory)
    final_report["health_check"] = health_check
    final_report["questions_processed"] = questions_processed

    logger.info("Комплексный анализ завершен")
    return final_report
