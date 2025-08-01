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
import urllib.parse

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
        dataframes: dict[str, pd.DataFrame], report_id: int
) -> dict[str, list[str]]:
    visualizations = {}
    sns.set_theme(style="whitegrid")

    for name, df in dataframes.items():
        if df.empty:
            logging.warning(f"DataFrame для таблицы '{name}' пуст, графики не будут сгенерированы.")
            continue

        # Prepare column info for GPT more explicitly, including types for better suggestions
        column_info = [
            {"name": col, "dtype": str(df[col].dtype), "nunique": df[col].nunique(),
             "is_numeric": pd.api.types.is_numeric_dtype(df[col])}
            for col in df.columns
        ]

        # Обновленный промпт с уточнением для bar-графиков и явным указанием типов столбцов
        prompt = (
            f"Проанализируй DataFrame с названием '{name}' со следующими столбцами и их характеристиками: {json.dumps(column_info)}. "
            "Предложи до 2 наиболее подходящих визуализаций для анализа этих данных. "
            "Ответ предоставь в виде JSON-объекта с ключом 'charts', который содержит массив предложений. "
            "Каждый объект в массиве должен содержать: 'chart_type' ('hist', 'bar', 'scatter', 'pie'), "
            "'columns' (СПИСОК СУЩЕСТВУЮЩИХ СТОЛБЦОВ, ВЫБРАННЫХ ИЗ ПРЕДОСТАВЛЕННОГО СПИСКА). Для 'bar' графика: если это распределение по одной категории (частота), используй 1 столбец; если это агрегированное значение по категории (например, сумма), используй 2 столбца (категория, числовой показатель). И 'title' (название графика на русском). "
            "Выбирай столбцы с умом. Не предлагай scatter если нет двух числовых колонок или pie для колонок с >10 уникальных значений. "
            "Пример: {{\"charts\": [{{\"chart_type\": \"bar\", \"columns\": [\"col1\"], \"title\": \"Пример распределения\"}}, {{\"chart_type\": \"bar\", \"columns\": [\"col1\", \"col2\"], \"title\": \"Пример суммы\"}}]}}"
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
                chart_type = idea.get("chart_type")
                columns = idea.get("columns", [])
                title = idea.get("title", "Сгенерированный график")

                # Ensure all proposed columns exist in the DataFrame
                if not all(col in df.columns for col in columns):
                    logging.warning(
                        f"Не удалось создать график '{title}' для таблицы '{name}': Один или несколько предложенных столбцов ({', '.join(columns)}) не найдены в DataFrame.")
                    plt.close()  # Important to close the figure
                    continue

                # Create a copy for plotting to avoid modifying the original dataframe in case of type conversion
                plot_df = df.copy()

                try:
                    if chart_type == 'hist' and len(columns) == 1:
                        col = columns[0]
                        # Attempt to convert to numeric, coerce errors to NaN, then drop NaNs for plotting
                        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                        if pd.api.types.is_numeric_dtype(plot_df[col]):
                            sns.histplot(plot_df.dropna(subset=[col]), x=col, kde=True)
                        else:
                            logging.warning(
                                f"Не удалось создать гистограмму для '{col}': Столбец не является числовым после попытки преобразования.")
                            plt.close()
                            continue
                    elif chart_type == 'bar':  # Bar chart can take 1 or 2 columns
                        if len(columns) == 1:
                            col = columns[0]
                            # For single-column bar chart (frequency distribution)
                            value_counts = plot_df[col].value_counts().nlargest(15)  # Top 15 categories
                            if not value_counts.empty:
                                sns.barplot(x=value_counts.index, y=value_counts.values)
                                plt.xticks(rotation=45, ha='right')
                                plt.ylabel('Частота')  # Add Y-axis label for frequency
                            else:
                                logging.warning(
                                    f"Недостаточно данных для построения столбчатой диаграммы частоты для '{col}'.")
                                plt.close()
                                continue
                        elif len(columns) == 2:
                            x_col, y_col = columns[0], columns[1]
                            # Attempt to convert y_col to numeric
                            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')

                            if pd.api.types.is_numeric_dtype(plot_df[y_col]):
                                # Filter out NaN in relevant columns for grouping/summing
                                temp_df = plot_df.dropna(subset=[x_col, y_col])
                                if not temp_df.empty:
                                    top_15 = temp_df.groupby(x_col)[y_col].sum().nlargest(15)
                                    sns.barplot(x=top_15.index, y=top_15.values)
                                    plt.xticks(rotation=45, ha='right')
                                else:
                                    logging.warning(
                                        f"Недостаточно данных для построения столбчатой диаграммы агрегации для '{x_col}' и '{y_col}' после очистки NaN.")
                                    plt.close()
                                    continue
                            else:
                                logging.warning(
                                    f"Не удалось создать столбчатую диаграмму агрегации для '{y_col}': Столбец агрегации не является числовым после попытки преобразования.")
                                plt.close()
                                continue
                        else:  # If column count is not 1 or 2 for bar chart
                            logging.warning(
                                f"Не удалось создать столбчатую диаграмму '{title}' для таблицы '{name}': Неверное количество столбцов ({len(columns)}). Ожидается 1 или 2.")
                            plt.close()
                            continue
                    elif chart_type == 'scatter' and len(columns) == 2:
                        x_col, y_col = columns[0], columns[1]
                        # Attempt to convert both columns to numeric
                        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
                        plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')

                        if pd.api.types.is_numeric_dtype(plot_df[x_col]) and pd.api.types.is_numeric_dtype(
                                plot_df[y_col]):
                            # Drop NaNs for scatter plot
                            sns.scatterplot(plot_df.dropna(subset=[x_col, y_col]), x=x_col, y=y_col)
                        else:
                            logging.warning(
                                f"Не удалось создать точечную диаграмму для '{x_col}' и '{y_col}': Один или оба "
                                f"столбца не являются числовыми после попытки преобразования.")
                            plt.close()
                            continue
                    elif chart_type == 'pie' and len(columns) == 1:
                        col = columns[0]
                        # For pie, ensure the column is suitable (e.g., categorical or low unique values)
                        if plot_df[col].nunique() <= 10 and not plot_df[
                            col].empty:  # Re-check nunique and ensure not empty
                            plot_df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False)
                            plt.ylabel('')  # Remove y-label for pie chart
                        else:
                            logging.warning(
                                f"Не удалось создать круговую диаграмму для '{col}': Слишком много уникальных "
                                f"значений (>10) или столбец пуст.")
                            plt.close()
                            continue
                    else:  # Fallback for unsupported chart types or invalid column counts
                        logging.warning(
                            f"Не удалось создать график '{title}' для таблицы '{name}': Неподдерживаемый тип графика ({chart_type}) или неверное количество столбцов ({len(columns)}).")
                        plt.close()
                        continue

                    plt.title(title)
                    plt.tight_layout()

                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight')
                    buffer.seek(0)

                    file_name_for_s3 = f"{urllib.parse.quote(name)}_{i}.png"
                    s3_key = f"charts/{report_id}/{file_name_for_s3}"

                    blob = config.gcs_bucket.blob(s3_key)
                    blob.upload_from_string(buffer.getvalue(), content_type='image/png')

                    blob = config.gcs_bucket.blob(s3_key)
                    presigned_url = blob.generate_signed_url(expiration=604800)
                    chart_urls.append(presigned_url)
                except Exception as e:
                    logging.error(
                        f"Не удалось создать график '{title}' (таблица: '{name}', тип: '{chart_type}', столбцы: {columns}): {e}",
                        exc_info=True)
                finally:
                    plt.close()  # Always close the plot to free memory

            if chart_urls:
                visualizations[name] = chart_urls

        except json.JSONDecodeError as jde:
            logging.error(
                f"Не удалось распарсить JSON от OpenAI для '{name}'. Ответ: {response.choices[0].message.content}. Ошибка: {jde}")
        except Exception as e:
            logging.error(f"Общая ошибка при генерации идей для графиков для '{name}': {e}", exc_info=True)

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
