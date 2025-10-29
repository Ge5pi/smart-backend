from typing import Dict, Any, Set, Tuple, List
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy.engine import Inspector
from openai import OpenAI
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
import config
from config import API_KEY
import urllib.parse
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import redis
import pickle
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

# Инициализация OpenAI клиента
client = OpenAI(api_key=API_KEY)

# Инициализация Redis для кеширования
try:
    from config import REDIS_URL

    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
except:
    redis_client = None
    logging.warning("Redis не доступен, кеширование отключено")

# Настройки оптимизации
try:
    from config import (
        MAX_DATAFRAME_ROWS,
        MAX_TABLES_DETAILED_ANALYSIS,
        MAX_JOINS_TO_ANALYZE,
        MAX_CHARTS_PER_TABLE,
        MAX_TABLES_TO_VISUALIZE,
        CACHE_TTL_ANALYSIS,
        CACHE_TTL_VISUALIZATIONS
    )
except ImportError:
    MAX_DATAFRAME_ROWS = 100000
    MAX_TABLES_DETAILED_ANALYSIS = 10
    MAX_JOINS_TO_ANALYZE = 10
    MAX_CHARTS_PER_TABLE = 2
    MAX_TABLES_TO_VISUALIZE = 5
    CACHE_TTL_ANALYSIS = 3600
    CACHE_TTL_VISUALIZATIONS = 7200


def get_df_hash(df: pd.DataFrame) -> str:
    """Создает хеш DataFrame для кеширования"""
    df_string = f"{df.shape}:{df.columns.tolist()}:{df.head().to_json()}"
    return hashlib.md5(df_string.encode()).hexdigest()


def analyze_single_table(table_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Анализирует одну таблицу с использованием GPT и статистических методов.
    Включает кеширование для повторных запросов.
    """
    # Проверяем кеш
    if redis_client:
        df_hash = get_df_hash(df)
        cache_key = f"analysis:{table_name}:{df_hash}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logging.info(f"Найден кеш для таблицы {table_name}")
                return pickle.loads(cached)
        except Exception as e:
            logging.warning(f"Ошибка чтения кеша: {e}")

    # Ограничиваем размер данных для анализа
    if len(df) > 100000:
        df_sample = df.sample(n=100000, random_state=42)
        logging.info(f"Таблица {table_name}: используем выборку из 100000 строк для анализа")
    else:
        df_sample = df

    numeric_df = df_sample.select_dtypes(include=np.number)

    # Оптимизация корреляции - только для топ колонок
    if not numeric_df.empty and numeric_df.shape[1] > 10:
        # Берем только 10 самых важных числовых колонок по вариации
        variances = numeric_df.var().sort_values(ascending=False)
        top_cols = variances.head(10).index.tolist()
        numeric_df = numeric_df[top_cols]

    corr = numeric_df.corr().replace({np.nan: None}).to_dict() if not numeric_df.empty else {}

    # Оптимизация статистики - только базовые метрики
    stats_dict = {}
    for col in df_sample.columns[:20]:  # Ограничиваем 20 колонками
        col_stats = {
            'count': int(df_sample[col].count()),
            'unique': int(df_sample[col].nunique())
        }
        if pd.api.types.is_numeric_dtype(df_sample[col]):
            col_stats.update({
                'mean': float(df_sample[col].mean()) if not pd.isna(df_sample[col].mean()) else None,
                'std': float(df_sample[col].std()) if not pd.isna(df_sample[col].std()) else None,
                'min': float(df_sample[col].min()) if not pd.isna(df_sample[col].min()) else None,
                'max': float(df_sample[col].max()) if not pd.isna(df_sample[col].max()) else None
            })
        stats_dict[col] = col_stats

    stats = json.dumps(stats_dict, ensure_ascii=False)

    prompt = (
        f"Проанализируй данные из таблицы '{table_name}'. "
        f"Строк в таблице: {len(df)}. "
        f"Вот описательная статистика (выборка): {stats}. "
        f"Вот матрица корреляций для ключевых числовых полей: {json.dumps(corr)}. "
        "Твоя задача — выявить ключевые инсайты, скрытые закономерности и аномалии в данных этой таблицы. "
        "Будь кратким, структурированным и пиши на русском языке, используя Markdown для выделения (`**термин**`)."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        insight = response.choices[0].message.content
    except Exception as e:
        logging.error(f"Ошибка при вызове GPT для таблицы {table_name}: {e}")
        insight = "Не удалось получить анализ от GPT"

    result = {"insight": insight, "correlations": corr, "row_count": len(df)}

    # Сохраняем в кеш
    if redis_client:
        try:
            redis_client.setex(cache_key, CACHE_TTL_ANALYSIS, pickle.dumps(result))
        except Exception as e:
            logging.warning(f"Ошибка записи в кеш: {e}")

    return result


def analyze_joins(inspector: Inspector, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Анализирует связи между таблицами через foreign keys.
    Оптимизирован для работы с ограниченным количеством связей.
    """
    joint_insights = {}
    analyzed_pairs: Set[Tuple[str, str]] = set()
    all_tables = list(dataframes.keys())
    join_count = 0

    for table_name in all_tables:
        if join_count >= MAX_JOINS_TO_ANALYZE:
            logging.info(f"Достигнут лимит анализа связей: {MAX_JOINS_TO_ANALYZE}")
            break

        try:
            foreign_keys = inspector.get_foreign_keys(table_name)
        except Exception as e:
            logging.warning(f"Не удалось получить внешние ключи для таблицы {table_name}: {e}")
            continue

        for fk in foreign_keys:
            if join_count >= MAX_JOINS_TO_ANALYZE:
                break

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
                # Ограничиваем размер перед JOIN
                max_rows_for_join = 5000
                df_left_sample = df_left.sample(n=min(len(df_left), max_rows_for_join), random_state=42)
                df_right_sample = df_right.sample(n=min(len(df_right), max_rows_for_join), random_state=42)

                merged_df = pd.merge(
                    df_left_sample, df_right_sample,
                    left_on=left_on,
                    right_on=right_on,
                    suffixes=(f'_{left_table}', f'_{right_table}'),
                    how='inner'  # Используем inner join для меньшего размера
                )

                if merged_df.empty or len(merged_df) < 10:
                    continue

                # Ограничиваем количество колонок для анализа
                if len(merged_df.columns) > 20:
                    # Берем только числовые колонки с наибольшей вариацией
                    numeric_cols = merged_df.select_dtypes(include=np.number).columns
                    if len(numeric_cols) > 10:
                        variances = merged_df[numeric_cols].var().sort_values(ascending=False)
                        top_numeric = variances.head(10).index.tolist()
                        other_cols = [c for c in merged_df.columns if c not in numeric_cols][:10]
                        merged_df = merged_df[top_numeric + other_cols]

                join_key = f"{left_table} 🔗 {right_table}"

                # Быстрая статистика без полного analyze_single_table
                numeric_df = merged_df.select_dtypes(include=np.number)
                corr = numeric_df.corr().replace({np.nan: None}).to_dict() if not numeric_df.empty else {}

                basic_stats = {
                    'merged_rows': len(merged_df),
                    'columns': len(merged_df.columns),
                    'left_rows': len(df_left),
                    'right_rows': len(df_right)
                }

                prompt = (
                    f"Проанализируй СВЯЗЬ между таблицами '{left_table}' ({basic_stats['left_rows']} строк) "
                    f"и '{right_table}' ({basic_stats['right_rows']} строк), которые были объединены "
                    f"по ключам ({left_table}.{','.join(left_on)} = {right_table}.{','.join(right_on)}). "
                    f"После объединения получилось {basic_stats['merged_rows']} строк. "
                    f"Ключевые корреляции: {json.dumps(corr)[:500]}. "
                    "Сосредоточься на поиске инсайтов, которые возникают именно из-за связи двух таблиц. "
                    "Ответ дай на русском языке, кратко и по делу (2-3 предложения), используя Markdown."
                )

                try:
                    response = client.chat.completions.create(
                        model="gpt-5-mini",
                        messages=[{"role": "user", "content": prompt}]
                    )

                    joint_insights[join_key] = {
                        "insight": response.choices[0].message.content,
                        "correlations": corr,
                        "stats": basic_stats
                    }
                except Exception as e:
                    logging.error(f"Ошибка GPT при анализе связи {join_key}: {e}")
                    joint_insights[join_key] = {
                        "insight": "Не удалось получить анализ связи",
                        "correlations": corr,
                        "stats": basic_stats
                    }

                join_count += 1

            except Exception as e:
                logging.error(f"Ошибка при объединении и анализе таблиц {left_table} и {right_table}: {e}")

    return joint_insights


def generate_visualizations(
        dataframes: dict[str, pd.DataFrame], report_id: int
) -> dict[str, list[str]]:
    """
    Генерирует визуализации для таблиц с использованием GPT и matplotlib.
    Включает кеширование и ограничения на количество графиков.
    """
    visualizations = {}
    sns.set_theme(style="whitegrid")

    # Ограничиваем количество графиков
    table_count = 0

    # Сортируем таблицы по размеру
    sorted_tables = sorted(dataframes.items(), key=lambda x: len(x[1]), reverse=True)

    for name, df in sorted_tables:
        if table_count >= MAX_TABLES_TO_VISUALIZE:
            logging.info(f"Достигнут лимит визуализаций: {MAX_TABLES_TO_VISUALIZE} таблиц")
            break

        if df.empty or len(df) < 10:
            logging.warning(f"DataFrame для таблицы '{name}' слишком мал, пропускаем визуализацию.")
            continue

        # Проверяем кеш визуализаций
        if redis_client:
            cache_key = f"viz:{report_id}:{name}"
            try:
                cached_urls = redis_client.get(cache_key)
                if cached_urls:
                    visualizations[name] = pickle.loads(cached_urls)
                    logging.info(f"Используем закешированные визуализации для {name}")
                    table_count += 1
                    continue
            except Exception as e:
                logging.warning(f"Ошибка чтения кеша визуализаций: {e}")

        column_info = [
            {"name": col, "dtype": str(df[col].dtype), "nunique": df[col].nunique(),
             "is_numeric": pd.api.types.is_numeric_dtype(df[col])}
            for col in df.columns
        ]

        prompt = (
            f"Проанализируй DataFrame с названием '{name}' со следующими столбцами и их характеристиками: "
            f"{json.dumps(column_info)}. "
            f"Предложи до {MAX_CHARTS_PER_TABLE} наиболее подходящих визуализаций для анализа этих данных. "
            "Ответ предоставь в виде JSON-объекта с ключом 'charts', который содержит массив предложений. "
            "Каждый объект в массиве должен содержать: 'chart_type' ('hist', 'bar', 'scatter', 'pie'), "
            "'columns' (СПИСОК СУЩЕСТВУЮЩИХ СТОЛБЦОВ, ВЫБРАННЫХ ИЗ ПРЕДОСТАВЛЕННОГО СПИСКА). Для 'bar' графика: если "
            "это распределение по одной категории (частота), используй 1 столбец; если это агрегированное значение по "
            "категории (например, сумма), используй 2 столбца (категория, числовой показатель). И 'title' (название "
            "графика на русском). "
            "Выбирай столбцы с умом. Не предлагай scatter если нет двух числовых колонок или pie для колонок с >10 "
            "уникальных значений. "
            "Пример: {\\\"charts\\\": [{\\\"chart_type\\\": \\\"bar\\\", \\\"columns\\\": [\\\"col1\\\"], \\\"title\\\": \\\"Пример "
            "распределения\\\"}, {\\\"chart_type\\\": \\\"bar\\\", \\\"columns\\\": [\\\"col1\\\", \\\"col2\\\"], \\\"title\\\": \\\"Пример "
            "суммы\\\"}]} "
        )

        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            response_data = json.loads(response.choices[0].message.content)
            chart_ideas = response_data.get("charts", [])

            if not chart_ideas:
                logging.warning(f"GPT не предложил идей для графиков для '{name}'.")
                continue

            # Ограничиваем количество графиков на таблицу
            chart_ideas = chart_ideas[:MAX_CHARTS_PER_TABLE]

            chart_urls = []
            for i, idea in enumerate(chart_ideas):
                plt.figure(figsize=(10, 6))
                chart_type = idea.get("chart_type")
                columns = idea.get("columns", [])
                title = idea.get("title", "Сгенерированный график")

                # Проверяем существование колонок
                if not all(col in df.columns for col in columns):
                    logging.warning(
                        f"Не удалось создать график '{title}' для таблицы '{name}': Один или несколько предложенных "
                        f"столбцов ({', '.join(columns)}) не найдены в DataFrame.")
                    plt.close()
                    continue

                plot_df = df.copy()

                try:
                    if chart_type == 'hist' and len(columns) == 1:
                        col = columns[0]
                        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
                        if pd.api.types.is_numeric_dtype(plot_df[col]):
                            sns.histplot(plot_df.dropna(subset=[col]), x=col, kde=True)
                        else:
                            logging.warning(
                                f"Не удалось создать гистограмму для '{col}': Столбец не является числовым после "
                                f"попытки преобразования.")
                            plt.close()
                            continue

                    elif chart_type == 'bar':
                        if len(columns) == 1:
                            col = columns[0]
                            value_counts = plot_df[col].value_counts().nlargest(15)
                            if not value_counts.empty:
                                sns.barplot(x=value_counts.index, y=value_counts.values)
                                plt.xticks(rotation=45, ha='right')
                                plt.ylabel('Частота')
                            else:
                                logging.warning(
                                    f"Недостаточно данных для построения столбчатой диаграммы частоты для '{col}'.")
                                plt.close()
                                continue

                        elif len(columns) == 2:
                            x_col, y_col = columns[0], columns[1]
                            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
                            if pd.api.types.is_numeric_dtype(plot_df[y_col]):
                                temp_df = plot_df.dropna(subset=[x_col, y_col])
                                if not temp_df.empty:
                                    top_15 = temp_df.groupby(x_col)[y_col].sum().nlargest(15)
                                    sns.barplot(x=top_15.index, y=top_15.values)
                                    plt.xticks(rotation=45, ha='right')
                                else:
                                    logging.warning(
                                        f"Недостаточно данных для построения столбчатой диаграммы агрегации для "
                                        f"'{x_col}' и '{y_col}' после очистки NaN.")
                                    plt.close()
                                    continue
                            else:
                                logging.warning(
                                    f"Не удалось создать столбчатую диаграмму агрегации для '{y_col}': Столбец "
                                    f"агрегации не является числовым после попытки преобразования.")
                                plt.close()
                                continue
                        else:
                            logging.warning(
                                f"Не удалось создать столбчатую диаграмму '{title}' для таблицы '{name}': Неверное "
                                f"количество столбцов ({len(columns)}). Ожидается 1 или 2.")
                            plt.close()
                            continue

                    elif chart_type == 'scatter' and len(columns) == 2:
                        x_col, y_col = columns[0], columns[1]
                        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors='coerce')
                        plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
                        if pd.api.types.is_numeric_dtype(plot_df[x_col]) and pd.api.types.is_numeric_dtype(
                                plot_df[y_col]):
                            sns.scatterplot(plot_df.dropna(subset=[x_col, y_col]), x=x_col, y=y_col)
                        else:
                            logging.warning(
                                f"Не удалось создать точечную диаграмму для '{x_col}' и '{y_col}': Один или оба "
                                f"столбца не являются числовыми после попытки преобразования.")
                            plt.close()
                            continue

                    elif chart_type == 'pie' and len(columns) == 1:
                        col = columns[0]
                        if plot_df[col].nunique() <= 10 and not plot_df[col].empty:
                            plot_df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, counterclock=False)
                            plt.ylabel('')
                        else:
                            logging.warning(
                                f"Не удалось создать круговую диаграмму для '{col}': Слишком много уникальных "
                                f"значений (>10) или столбец пуст.")
                            plt.close()
                            continue

                    else:
                        logging.warning(
                            f"Не удалось создать график '{title}' для таблицы '{name}': Неподдерживаемый тип графика"
                            f" ({chart_type}) или неверное количество столбцов ({len(columns)}).")
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

                    api_url = f"{config.BASE_API_URL}/chart/{report_id}/{file_name_for_s3}"
                    chart_urls.append(api_url)

                except Exception as e:
                    logging.error(
                        f"Не удалось создать график '{title}' (таблица: '{name}', тип: '{chart_type}', столбцы: {columns}): {e}",
                        exc_info=True)
                finally:
                    plt.close()

            if chart_urls:
                visualizations[name] = chart_urls

                # Сохраняем в кеш
                if redis_client:
                    try:
                        redis_client.setex(f"viz:{report_id}:{name}", CACHE_TTL_VISUALIZATIONS,
                                           pickle.dumps(chart_urls))
                    except Exception as e:
                        logging.warning(f"Ошибка записи визуализаций в кеш: {e}")

            table_count += 1

        except json.JSONDecodeError as jde:
            logging.error(
                f"Не удалось распарсить JSON от OpenAI для '{name}'. Ответ: {response.choices[0].message.content}. "
                f"Ошибка: {jde}")
        except Exception as e:
            logging.error(f"Общая ошибка при генерации идей для графиков для '{name}': {e}", exc_info=True)

    return visualizations


def cluster_data(df: pd.DataFrame, table_name: str, n_clusters: int = 3) -> dict:
    """
    Выполняет кластеризацию данных с использованием KMeans
    """
    numeric_df = df.select_dtypes(include=np.number).dropna()

    if numeric_df.shape[0] < 10 or numeric_df.shape[1] < 2:
        return {"message": "Недостаточно данных для кластеризации"}

    try:
        # Стандартизация данных
        scaled = StandardScaler().fit_transform(numeric_df)

        # KMeans кластеризация
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        kmeans.fit(scaled)

        # Размеры кластеров
        sizes = pd.Series(kmeans.labels_).value_counts().sort_index().to_dict()

        # Профили кластеров
        cluster_profiles_df = numeric_df.copy()
        cluster_profiles_df["cluster"] = kmeans.labels_
        profiles = cluster_profiles_df.groupby("cluster").mean().round(3).to_dict()

        # Важность признаков
        rf = RandomForestClassifier(random_state=42, n_estimators=50)  # Снижено для скорости
        rf.fit(scaled, kmeans.labels_)
        importances = rf.feature_importances_
        feature_importance = sorted(
            zip(numeric_df.columns, importances),
            key=lambda x: x[1],
            reverse=True
        )

        top_features = [f[0] for f in feature_importance[:5]]

        prompt = (
            f"Мы провели кластеризацию таблицы '{table_name}' на {n_clusters} кластера(ов). "
            f"Размеры кластеров: {sizes}. "
            f"Вот средние значения признаков по кластерам: {json.dumps(profiles, ensure_ascii=False)}. "
            f"Важнейшие признаки для разделения: {top_features}. "
            "Дай понятное описание каждого кластера и предложи, что это могут быть за группы. "
            "Ответь кратко, на русском, в Markdown."
        )

        try:
            gpt_response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            gpt_summary = gpt_response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Ошибка GPT при описании кластеров: {e}")
            gpt_summary = None

        return {
            "clusters_count": n_clusters,
            "sample_count": numeric_df.shape[0],
            "feature_count": numeric_df.shape[1],
            "sizes": sizes,
            "cluster_profiles": profiles,
            "important_features": top_features,
            "gpt_summary": gpt_summary
        }

    except Exception as e:
        return {"error": str(e)}


def generate_and_test_hypotheses(df: pd.DataFrame, table_name: str) -> List[Dict[str, Any]]:
    """
    Генерирует и проверяет статистические гипотезы для таблицы
    """
    hypotheses_results = []

    # Минимальные данные для проверки
    if df.shape[1] < 2:
        logging.info(f"Таблица {table_name}: слишком мало колонок для генерации гипотез.")
        return []

    stats = df.describe(include='all').replace({pd.NA: None}).to_json()
    columns_info = [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns]

    prompt = (
        f"Вот DataFrame из таблицы '{table_name}', вот его статистика: {stats}. "
        f"Вот столбцы: {columns_info}. "
        "Сформулируй 2 гипотезы, которые можно проверить статистически, "
        "и укажи, по каким столбцам их проверять. "
        "Ответь строго в формате JSON массива объектов, без лишнего текста: "
        "[{\\\"hypothesis\\\": \\\"...\\\", \\\"test\\\": \\\"t-test\\\" или \\\"chi2\\\", \\\"columns\\\": [\\\"col1\\\", \\\"col2\\\"]}]."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        raw_output = response.choices[0].message.content.strip()
        logging.debug(f"GPT hypothesis raw output for {table_name}: {raw_output}")

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logging.warning(f"Не удалось распарсить JSON гипотез для {table_name}: {e}")
            return []

        if not isinstance(parsed, list):
            logging.warning(f"Неверный формат ответа GPT для {table_name}, ожидался список: {parsed}")
            return []

        for item in parsed:
            if not isinstance(item, dict):
                logging.warning(f"Неверный элемент в списке гипотез для {table_name}: {item}")
                continue

            hypothesis = item.get("hypothesis", "").strip()
            test = item.get("test", "").strip().lower()
            cols = item.get("columns", [])

            if not hypothesis or not test or not isinstance(cols, list) or len(cols) < 2:
                logging.warning(f"Неполная гипотеза для {table_name}: {item}")
                continue

            explanation = ""
            p_value = None
            result = "не удалось проверить"

            try:
                if test == "t-test":
                    group_col, value_col = cols[0], cols[1]
                    if group_col in df.columns and value_col in df.columns:
                        groups = df[group_col].dropna().unique()
                        if len(groups) == 2:
                            a = df[df[group_col] == groups[0]][value_col].dropna()
                            b = df[df[group_col] == groups[1]][value_col].dropna()
                            if len(a) > 1 and len(b) > 1:
                                stat, p = ttest_ind(a, b)
                                result = "подтверждена" if p < 0.05 else "опровергнута"
                                p_value = round(float(p), 5)
                                explanation = f"t-test между {groups[0]} и {groups[1]} по {value_col}"

                elif test == "chi2":
                    col_a, col_b = cols[0], cols[1]
                    if col_a in df.columns and col_b in df.columns:
                        contingency = pd.crosstab(df[col_a], df[col_b])
                        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                            stat, p, *_ = chi2_contingency(contingency)
                            result = "подтверждена" if p < 0.05 else "опровергнута"
                            p_value = round(float(p), 5)
                            explanation = f"chi2 между {col_a} и {col_b}"

            except Exception as e:
                explanation = f"Ошибка при проверке: {e}"
                logging.warning(f"Ошибка при проверке гипотезы {hypothesis} в {table_name}: {e}")

            hypotheses_results.append({
                "hypothesis": hypothesis,
                "test": test,
                "columns": cols,
                "p_value": p_value,
                "result": result,
                "explanation": explanation
            })

    except Exception as e:
        logging.warning(f"Не удалось сгенерировать или проверить гипотезы для {table_name}: {e}")

    return hypotheses_results


def perform_full_analysis(
        inspector: Inspector, dataframes: Dict[str, pd.DataFrame], report_id: int
) -> Dict[str, Any]:
    """
    Выполняет полный анализ базы данных с оптимизациями производительности.
    Включает параллельную обработку и приоритизацию таблиц.
    """
    start_time = time.time()

    # Профилирование функции
    def log_step(step_name, step_start):
        duration = time.time() - step_start
        logging.info(f"[Отчет {report_id}] {step_name}: {duration:.2f}s")
        return time.time()

    # Фильтруем только топ-N самых важных таблиц
    table_sizes = {name: len(df) for name, df in dataframes.items()}
    sorted_tables = sorted(table_sizes.items(), key=lambda x: x[1], reverse=True)

    # Анализируем подробно только топ таблиц, остальные - базовый анализ
    priority_tables = [t[0] for t in sorted_tables[:MAX_TABLES_DETAILED_ANALYSIS]]

    logging.info(f"Приоритетные таблицы для анализа: {priority_tables}")

    # Параллельный анализ отдельных таблиц
    step_start = time.time()
    single_table_analysis = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_table = {
            executor.submit(analyze_single_table, table, df): table
            for table, df in dataframes.items() if table in priority_tables
        }

        for future in future_to_table:
            table = future_to_table[future]
            try:
                single_table_analysis[table] = future.result(timeout=120)  # 2 минуты на таблицу
            except Exception as e:
                logging.error(f"Ошибка анализа таблицы {table}: {e}")
                single_table_analysis[table] = {"insight": "Ошибка анализа", "correlations": {}}

    # Для остальных таблиц - минимальный анализ без GPT
    for table, df in dataframes.items():
        if table not in priority_tables:
            numeric_df = df.select_dtypes(include=np.number)
            corr = numeric_df.corr().replace({np.nan: None}).to_dict() if not numeric_df.empty else {}
            single_table_analysis[table] = {
                "insight": f"Базовая информация: {len(df)} строк, {len(df.columns)} колонок",
                "correlations": corr,
                "row_count": len(df)
            }

    step_start = log_step("Анализ отдельных таблиц", step_start)

    # Анализ связей между таблицами
    joint_table_analysis = analyze_joins(inspector, dataframes)
    step_start = log_step("Анализ связей", step_start)

    # Визуализация только для приоритетных таблиц
    priority_dataframes = {k: v for k, v in dataframes.items() if k in priority_tables[:MAX_TABLES_TO_VISUALIZE]}
    visualizations = generate_visualizations(priority_dataframes.copy(), report_id)
    step_start = log_step("Генерация визуализаций", step_start)

    # Генерация общего резюме
    def generate_overall_summary(dataframes, insights, joins):
        prompt = f"""
Ты — аналитик. Сделай строгий обзор БД на русском в Markdown без преамбул и без «если нужно, могу…».

Дано:
- Таблицы: {list(dataframes.keys())}
- Инсайты: {json.dumps(insights, ensure_ascii=False)[:5000]}
- Связи: {json.dumps(joins, ensure_ascii=False)[:3000]}

Структура ответа (ровно эти секции и порядок):

# Общий обзор

## Ключевые тренды

## Связи между таблицами

## Аномалии и выбросы

## Повторяющиеся паттерны

## Риски и ограничения анализа

## Что проверить дополнительно

Правила:
- Только по данным выше, без выдумок. Привязывай выводы к таблицам/полям/инсайтам.
- В трендах/связях указывай возможные причины и альтернативы (если есть).
- В аномалиях — где видно и возможная природа (ошибка/сезонность/редкость).
- В рисках — конкретные недостатки данных и неоднозначности.
- В «Что проверить дополнительно» — 3–7 конкретных проверок, без SQL, без просьб/предложений.
- Тон сухой, аналитический. Без эмодзи, без call-to-action, без «я могу/готов/предлагаю».

Запрещено:
- Любые сервисные фразы до/после секций, предложения помощи, планы, SQL, шаблоны.
- Любые выводы, не подтвержденные исходной информацией.

Выход: только Markdown с указанными секциями.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Ошибка при генерации общего резюме: {e}")
            return "Не удалось сгенерировать общее резюме"

    overall_summary = generate_overall_summary(dataframes, single_table_analysis, joint_table_analysis)
    step_start = log_step("Генерация общего резюме", step_start)

    # Генерация и проверка гипотез (только для приоритетных таблиц)
    hypotheses_by_table = {}
    for table in priority_tables[:5]:  # Только топ-5 таблиц
        df = dataframes[table]
        hypotheses_by_table[table] = generate_and_test_hypotheses(df, table)
    step_start = log_step("Генерация гипотез", step_start)

    # Кластеризация (только для приоритетных таблиц)
    clusters_by_table = {}
    for table in priority_tables[:5]:  # Только топ-5 таблиц
        df = dataframes[table]
        clusters_by_table[table] = cluster_data(df, table)
    step_start = log_step("Кластеризация", step_start)

    # Очистка NaN значений
    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_nan(item) for item in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    total_duration = time.time() - start_time
    logging.info(f"[Отчет {report_id}] Полный анализ завершен за {total_duration:.2f}s")

    return clean_nan({
        "single_table_insights": single_table_analysis,
        "joint_table_insights": joint_table_analysis,
        "visualizations": visualizations,
        "overall_summary": overall_summary,
        "hypotheses": hypotheses_by_table,
        "clusters": clusters_by_table
    })
