from typing import Dict, Any, Set, Tuple, List
import pandas as pd
import numpy as np
import io
from sqlalchemy.engine import Inspector
from openai import OpenAI
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
import config
from config import API_KEY

client = OpenAI(api_key=API_KEY)


def analyze_single_table(table_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr().replace({np.nan: None}).to_dict() if not numeric_df.empty else {}
    stats = df.describe(include='all').replace({np.nan: None}).to_json()

    prompt = (
        f"Проанализируй данные из таблицы '{table_name}'. "
        f"Вот описательная статистика: {stats}. "
        f"Вот матрица корреляций для числовых полей: {json.dumps(corr)}. "
        "Твоя задача — выявить ключевые инсайты, скрытые закономерности и аномалии в данных этой таблицы. "
        "Будь кратким, структурированным и пиши на русском языке, используя Markdown для выделения (`**термин**`)."
    )

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    insight = response.choices[0].message.content
    return {"insight": insight, "correlations": corr}


def analyze_joins(inspector: Inspector, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    joint_insights = {}
    analyzed_pairs: Set[Tuple[str, str]] = set()
    all_tables = list(dataframes.keys())

    for table_name in all_tables:
        try:
            foreign_keys = inspector.get_foreign_keys(table_name)
        except Exception as e:
            logging.warning(f"Не удалось получить внешние ключи для таблицы {table_name}: {e}")
            continue

        for fk in foreign_keys:
            left_table = table_name
            right_table = fk['referred_table']

            pair = tuple(sorted((left_table, right_table)))
            if pair in analyzed_pairs:
                continue

            analyzed_pairs.add(pair)

            df_left = dataframes[left_table]
            df_right = dataframes.get(right_table)

            if df_right is None:
                continue

            left_on = fk['constrained_columns']
            right_on = fk['referred_columns']

            try:
                merged_df = pd.merge(
                    df_left, df_right,
                    left_on=left_on,
                    right_on=right_on,
                    suffixes=(f'_{left_table}', f'_{right_table}')
                )

                if merged_df.empty:
                    continue

                join_key = f"{left_table} 🔗 {right_table}"
                analysis_result = analyze_single_table(join_key, merged_df)
                stats = merged_df.describe(include='all').replace({np.nan: None}).to_json()
                corr = analysis_result["correlations"]

                prompt = (
                    f"Проанализируй СВЯЗЬ между таблицами '{left_table}' и '{right_table}', которые были объединены по ключам "
                    f"({left_table}.{','.join(left_on)} = {right_table}.{','.join(right_on)}). "
                    f"Вот статистика объединенных данных: {stats}. "
                    f"Вот матрица корреляций: {json.dumps(corr)}. "
                    "Сосредоточься на поиске инсайтов, которые возникают именно из-за связи двух таблиц. "
                    "Ответ дай на русском языке, кратко и по делу, используя Markdown для выделения (`**термин**`)."
                )

                response = client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5
                )

                analysis_result["insight"] = response.choices[0].message.content
                joint_insights[join_key] = analysis_result

            except Exception as e:
                logging.error(f"Ошибка при объединении и анализе таблиц {left_table} и {right_table}: {e}")

    return joint_insights


def generate_visualizations(
        dataframes: Dict[str, pd.DataFrame], report_id: int
) -> Dict[str, List[str]]:
    visualizations = {}
    sns.set_theme(style="whitegrid")

    for name, df in dataframes.items():
        if df.empty:
            continue

        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        df_info = df.dtypes.to_string()

        # ИЗМЕНЕНИЕ: Обновляем промпт для соответствия JSON Mode
        prompt = (
            f"Для DataFrame с названием '{name}' и следующей структурой столбцов:\n{df_info}\n\n"
            "Предложи до 2 наиболее подходящих визуализаций для анализа этих данных. "
            "Ответ предоставь в виде JSON-объекта с ключом 'charts', который содержит массив предложений. "
            "Каждый объект в массиве должен содержать: 'chart_type' ('hist', 'bar', 'scatter', 'pie'), "
            "'columns' (список столбцов), и 'title' (название графика на русском). "
            "Выбирай столбцы с умом. Не предлагай scatter если нет двух числовых колонок или pie для колонок с >10 уникальных значений. "
            "Пример: {\"charts\": [{\"chart_type\": \"bar\", \"columns\": [\"col1\", \"col2\"], \"title\": \"Пример\"}]}"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            response_data = json.loads(response.choices[0].message.content)
            chart_ideas = response_data.get("charts", [])

            if not chart_ideas:
                logging.warning(f"GPT не предложил идей для графиков для '{name}'.")
                continue

            chart_urls = []
            for i, idea in enumerate(chart_ideas):
                plt.figure(figsize=(10, 6))
                chart_type, columns, title = idea.get("chart_type"), idea.get("columns", []), idea.get("title",
                                                                                                       "Сгенерированный график")

                try:
                    if chart_type == 'hist' and len(columns) == 1 and pd.api.types.is_numeric_dtype(df[columns[0]]):
                        sns.histplot(df, x=columns[0], kde=True)
                    elif chart_type == 'bar' and len(columns) == 2:
                        top_15 = df.groupby(columns[0])[columns[1]].sum().nlargest(15)
                        sns.barplot(x=top_15.index, y=top_15.values)
                        plt.xticks(rotation=45, ha='right')
                    elif chart_type == 'scatter' and len(columns) == 2 and all(
                            pd.api.types.is_numeric_dtype(df[c]) for c in columns):
                        sns.scatterplot(df, x=columns[0], y=columns[1])
                    elif chart_type == 'pie' and len(columns) == 1 and df[columns[0]].nunique() <= 10:
                        df[columns[0]].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False)
                        plt.ylabel('')
                    else:
                        continue

                    plt.title(title)
                    plt.tight_layout()

                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight')
                    buffer.seek(0)

                    s3_key = f"charts/{report_id}/{safe_name}_{i}.png"

                    config.s3_client.put_object(
                        Bucket=config.S3_BUCKET_NAME, Key=s3_key, Body=buffer, ContentType='image/png'
                    )

                    presigned_url = config.s3_client.generate_presigned_url(
                        'get_object', Params={'Bucket': config.S3_BUCKET_NAME, 'Key': s3_key}, ExpiresIn=360000000
                    )
                    chart_urls.append(presigned_url)
                except Exception as e:
                    logging.error(f"Не удалось создать график '{title}': {e}", exc_info=True)
                finally:
                    plt.close()

            if chart_urls:
                visualizations[name] = chart_urls

        except json.JSONDecodeError:
            logging.error(
                f"Не удалось распарсить JSON от OpenAI для '{name}'. Ответ: {response.choices[0].message.content}")
        except Exception as e:
            logging.error(f"Общая ошибка при генерации идей для графиков для '{name}': {e}")

    return visualizations


def perform_full_analysis(
        inspector: Inspector, dataframes: Dict[str, pd.DataFrame], report_id: int
) -> Dict[str, Any]:
    single_table_analysis = {}
    for table, df in dataframes.items():
        single_table_analysis[table] = analyze_single_table(table, df)

    joint_table_analysis = analyze_joins(inspector, dataframes)
    visualizations = generate_visualizations(dataframes.copy(), report_id)

    def clean_nan(obj):
        if isinstance(obj, dict): return {k: clean_nan(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean_nan(item) for item in obj]
        if isinstance(obj, float) and np.isnan(obj): return None
        return obj

    return clean_nan({
        "single_table_insights": single_table_analysis,
        "joint_table_insights": joint_table_analysis,
        "visualizations": visualizations
    })