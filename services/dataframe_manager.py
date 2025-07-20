# services/dataframe_manager.py
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from sqlalchemy import inspect, text
import numpy as np
import sys
import os

# Добавляем путь для импорта utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_serializer import convert_to_serializable, clean_dataframe_for_json

logger = logging.getLogger(__name__)


@dataclass
class TableRelation:
    """Описание связи между таблицами"""
    from_table: str
    to_table: str
    from_column: str
    to_column: str
    relation_type: str  # one_to_many, many_to_many, one_to_one


class DataFrameManager:
    """Менеджер для работы с связанными DataFrame"""

    def __init__(self, engine):
        self.engine = engine
        self.tables: Dict[str, pd.DataFrame] = {}
        self.relations: List[TableRelation] = []
        self.table_schemas: Dict[str, Dict] = {}
        self.total_memory_usage = 0.0

    def load_all_tables(self, max_rows_per_table: int = 100000) -> Dict[str, pd.DataFrame]:
        """Загружает все таблицы в память как DataFrame"""
        inspector = inspect(self.engine)
        tables_loaded = 0

        logger.info("Начинаем загрузку всех таблиц в DataFrame...")

        for table_name in inspector.get_table_names():
            if table_name in ['alembic_version', 'django_migrations', 'schema_migrations']:
                continue

            try:
                # Проверяем размер таблицы
                with self.engine.connect() as conn:
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.scalar()

                if row_count == 0:
                    logger.info(f"Пропускаем пустую таблицу {table_name}")
                    continue

                # Ограничиваем количество строк для больших таблиц
                if row_count > max_rows_per_table:
                    logger.warning(
                        f"Таблица {table_name} содержит {row_count} строк, загружаем только {max_rows_per_table}")
                    query = f"SELECT * FROM {table_name} LIMIT {max_rows_per_table}"
                else:
                    query = f"SELECT * FROM {table_name}"

                # Загружаем данные
                df = pd.read_sql(query, self.engine)

                if not df.empty:
                    # Обрабатываем проблемные типы данных
                    df = self._clean_dataframe(df)

                    self.tables[table_name] = df

                    # Сохраняем схему таблицы
                    columns = inspector.get_columns(table_name)
                    primary_keys = []
                    try:
                        pk_constraint = inspector.get_pk_constraint(table_name)
                        primary_keys = pk_constraint.get('constrained_columns', []) if pk_constraint else []
                    except:
                        primary_keys = []

                    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

                    self.table_schemas[table_name] = {
                        'columns': {col['name']: str(col['type']) for col in columns},
                        'row_count': len(df),
                        'total_row_count': int(row_count),
                        'primary_keys': primary_keys,
                        'memory_usage_mb': round(memory_mb, 3),
                        'is_truncated': row_count > max_rows_per_table
                    }

                    tables_loaded += 1
                    self.total_memory_usage += memory_mb

                    logger.info(f"✅ Загружена таблица {table_name}: {len(df)}/{row_count} строк, {memory_mb:.2f} MB")

            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {table_name}: {e}")

        # Анализируем связи между таблицами
        self._discover_relations(inspector)

        logger.info(f"🎉 Загружено {tables_loaded} таблиц, {len(self.relations)} связей, "
                    f"общий объем: {self.total_memory_usage:.2f} MB")

        return self.tables

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очищает DataFrame от проблемных значений"""
        try:
            df_clean = df.copy()

            # Заменяем inf на NaN
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

            # Обрабатываем проблемные типы данных
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    # Преобразуем None в пустые строки для объектных колонок
                    df_clean[col] = df_clean[col].astype(str).replace('None', '').replace('nan', '')
                elif pd.api.types.is_numeric_dtype(df_clean[col]):
                    # Для числовых колонок заменяем NaN на 0 если много пропусков
                    null_pct = df_clean[col].isnull().sum() / len(df_clean)
                    if null_pct > 0.5:  # Если больше 50% пропусков
                        df_clean[col] = df_clean[col].fillna(0)
                elif df_clean[col].dtype.kind in ['M', 'm']:  # datetime/timedelta
                    # Конвертируем datetime в строки для безопасности
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                        df_clean[col + '_str'] = df_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass

            return df_clean

        except Exception as e:
            logger.error(f"Ошибка очистки DataFrame: {e}")
            return df

    def _discover_relations(self, inspector):
        """Автоматически обнаруживает связи между таблицами"""
        relations_found = 0

        for table_name in self.tables.keys():
            try:
                foreign_keys = inspector.get_foreign_keys(table_name)
                for fk in foreign_keys:
                    referred_table = fk.get('referred_table')
                    if referred_table and referred_table in self.tables:
                        constrained_columns = fk.get('constrained_columns', [])
                        referred_columns = fk.get('referred_columns', [])

                        if constrained_columns and referred_columns:
                            relation = TableRelation(
                                from_table=table_name,
                                to_table=referred_table,
                                from_column=constrained_columns[0],
                                to_column=referred_columns[0],
                                relation_type='many_to_one'
                            )
                            self.relations.append(relation)
                            relations_found += 1

            except Exception as e:
                logger.warning(f"Не удалось получить FK для {table_name}: {e}")

        # Дополнительный поиск связей по именам колонок
        self._discover_implicit_relations()

        logger.info(f"Обнаружено {len(self.relations)} связей между таблицами")

    def _discover_implicit_relations(self):
        """Поиск неявных связей по именам колонок"""
        table_names = list(self.tables.keys())

        for i, table1 in enumerate(table_names):
            for table2 in table_names[i + 1:]:
                df1 = self.tables[table1]
                df2 = self.tables[table2]

                # Ищем общие колонки, которые могут быть связями
                common_columns = set(df1.columns).intersection(set(df2.columns))

                for col in common_columns:
                    if col.lower().endswith('_id') or col.lower() == 'id':
                        # Проверяем, есть ли пересечения значений
                        try:
                            values1 = set(df1[col].dropna().astype(str))
                            values2 = set(df2[col].dropna().astype(str))

                            if len(values1.intersection(values2)) > 0:
                                # Проверяем, что такой связи еще нет
                                existing = any(
                                    (r.from_table == table1 and r.to_table == table2 and r.from_column == col) or
                                    (r.from_table == table2 and r.to_table == table1 and r.to_column == col)
                                    for r in self.relations
                                )

                                if not existing:
                                    relation = TableRelation(
                                        from_table=table1,
                                        to_table=table2,
                                        from_column=col,
                                        to_column=col,
                                        relation_type='implicit'
                                    )
                                    self.relations.append(relation)
                        except:
                            continue

    def join_tables(self, left_table: str, right_table: str,
                    join_type: str = 'inner', on: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Выполняет JOIN между таблицами"""

        if left_table not in self.tables or right_table not in self.tables:
            raise ValueError(f"Таблицы {left_table} или {right_table} не найдены")

        left_df = self.tables[left_table].copy()
        right_df = self.tables[right_table].copy()

        # Автоматически определяем связь если не указана
        if on is None:
            relation = self._find_relation(left_table, right_table)
            if relation:
                on = {relation.from_column: relation.to_column}
            else:
                # Пытаемся найти общие колонки
                common_cols = set(left_df.columns).intersection(set(right_df.columns))
                if common_cols:
                    # Приоритет ID колонкам
                    id_cols = [col for col in common_cols if 'id' in col.lower()]
                    common_col = id_cols[0] if id_cols else list(common_cols)[0]
                    on = {common_col: common_col}
                else:
                    raise ValueError(f"Связь между {left_table} и {right_table} не найдена")

        try:
            # Выполняем JOIN
            result = pd.merge(
                left_df, right_df,
                left_on=list(on.keys()),
                right_on=list(on.values()),
                how=join_type,
                suffixes=('_left', '_right')
            )

            logger.info(f"JOIN выполнен: {left_table} {join_type} {right_table} на {on}, "
                        f"результат: {len(result)} строк")

            return result

        except Exception as e:
            logger.error(f"Ошибка JOIN: {e}")
            raise

    def _find_relation(self, table1: str, table2: str) -> Optional[TableRelation]:
        """Находит связь между двумя таблицами"""
        # Прямая связь
        for relation in self.relations:
            if relation.from_table == table1 and relation.to_table == table2:
                return relation

        # Обратная связь
        for relation in self.relations:
            if relation.from_table == table2 and relation.to_table == table1:
                return TableRelation(
                    from_table=table1,
                    to_table=table2,
                    from_column=relation.to_column,
                    to_column=relation.from_column,
                    relation_type=relation.relation_type
                )

        return None

    def complex_query(self, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Выполняет комплексный запрос с несколькими операциями"""
        result_df = None

        for i, op in enumerate(operations):
            op_type = op.get('type')
            logger.info(f"Выполнение операции {i + 1}: {op_type}")

            try:
                if op_type == 'select':
                    table_name = op['table']
                    columns = op.get('columns', None)

                    if table_name not in self.tables:
                        raise ValueError(f"Таблица {table_name} не найдена")

                    result_df = self.tables[table_name].copy()

                    if columns:
                        # Проверяем существование колонок
                        missing_cols = [col for col in columns if col not in result_df.columns]
                        if missing_cols:
                            logger.warning(f"Колонки не найдены: {missing_cols}")
                            columns = [col for col in columns if col in result_df.columns]

                        if columns:
                            result_df = result_df[columns]

                elif op_type == 'join':
                    if result_df is None:
                        raise ValueError("JOIN требует предварительного SELECT")

                    right_table = op['table']
                    join_type = op.get('how', 'inner')
                    on = op.get('on')

                    if right_table not in self.tables:
                        raise ValueError(f"Таблица {right_table} не найдена")

                    right_df = self.tables[right_table]

                    if on:
                        result_df = pd.merge(
                            result_df, right_df,
                            left_on=on['left'],
                            right_on=on['right'],
                            how=join_type,
                            suffixes=('', '_right')
                        )
                    else:
                        # Автоопределение связи
                        common_cols = set(result_df.columns).intersection(set(right_df.columns))
                        if common_cols:
                            # Приоритет ID колонкам
                            id_cols = [col for col in common_cols if 'id' in col.lower()]
                            common_col = id_cols[0] if id_cols else list(common_cols)[0]
                            result_df = pd.merge(
                                result_df, right_df,
                                on=common_col,
                                how=join_type,
                                suffixes=('', '_right')
                            )
                        else:
                            raise ValueError(f"Не найдены общие колонки для JOIN с {right_table}")

                elif op_type == 'filter':
                    if result_df is None:
                        raise ValueError("FILTER требует предварительного SELECT")

                    condition = op['condition']
                    try:
                        result_df = result_df.query(condition)
                    except Exception as filter_error:
                        logger.error(f"Ошибка фильтрации '{condition}': {filter_error}")
                        # Пытаемся выполнить простую фильтрацию
                        if 'column' in op and 'value' in op:
                            col = op['column']
                            val = op['value']
                            if col in result_df.columns:
                                result_df = result_df[result_df[col] == val]

                elif op_type == 'groupby':
                    if result_df is None:
                        raise ValueError("GROUPBY требует предварительного SELECT")

                    by = op['by']
                    agg_dict = op.get('agg', {})

                    # Проверяем существование колонок группировки
                    if isinstance(by, str):
                        by = [by]

                    existing_by = [col for col in by if col in result_df.columns]
                    if not existing_by:
                        logger.warning(f"Колонки для группировки не найдены: {by}")
                        continue

                    # Выполняем группировку
                    if agg_dict:
                        # Проверяем существование колонок для агрегации
                        valid_agg = {k: v for k, v in agg_dict.items() if k in result_df.columns}
                        if valid_agg:
                            result_df = result_df.groupby(existing_by).agg(valid_agg).reset_index()
                    else:
                        # Группировка со счетчиком
                        result_df = result_df.groupby(existing_by).size().reset_index(name='count')

                elif op_type == 'sort':
                    if result_df is None:
                        raise ValueError("SORT требует предварительного SELECT")

                    by = op['by']
                    ascending = op.get('ascending', True)

                    if isinstance(by, str):
                        by = [by]

                    existing_by = [col for col in by if col in result_df.columns]
                    if existing_by:
                        result_df = result_df.sort_values(existing_by, ascending=ascending)

                elif op_type == 'limit':
                    if result_df is None:
                        raise ValueError("LIMIT требует предварительного SELECT")

                    n = op['n']
                    result_df = result_df.head(n)

            except Exception as op_error:
                logger.error(f"Ошибка выполнения операции {op_type}: {op_error}")
                if result_df is None:
                    result_df = pd.DataFrame()
                continue

        if result_df is None:
            result_df = pd.DataFrame()

        logger.info(f"Комплексный запрос завершен, результат: {len(result_df)} строк")
        return result_df

    def get_table_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по загруженным таблицам"""
        return {
            'total_tables': len(self.tables),
            'total_relations': len(self.relations),
            'total_memory_mb': round(self.total_memory_usage, 2),
            'tables': {
                name: {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
                    'schema_info': self.table_schemas.get(name, {})
                }
                for name, df in self.tables.items()
            },
            'relations': [
                {
                    'from_table': rel.from_table,
                    'to_table': rel.to_table,
                    'from_column': rel.from_column,
                    'to_column': rel.to_column,
                    'type': rel.relation_type
                }
                for rel in self.relations
            ]
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """Возвращает детальную информацию об использовании памяти"""
        memory_info = {}

        for table_name, df in self.tables.items():
            memory_usage = df.memory_usage(deep=True)
            memory_info[table_name] = {
                'total_mb': round(memory_usage.sum() / 1024 / 1024, 3),
                'per_column_mb': {
                    col: round(usage / 1024 / 1024, 3)
                    for col, usage in memory_usage.items()
                }
            }

        return memory_info

    def optimize_memory(self):
        """Оптимизирует использование памяти"""
        for table_name, df in self.tables.items():
            try:
                original_size = df.memory_usage(deep=True).sum()

                # Оптимизируем числовые типы
                for col in df.select_dtypes(include=[np.number]).columns:
                    col_min = df[col].min()
                    col_max = df[col].max()

                    if pd.api.types.is_integer_dtype(df[col]):
                        if col_min >= 0:  # unsigned
                            if col_max < 255:
                                df[col] = df[col].astype(np.uint8)
                            elif col_max < 65535:
                                df[col] = df[col].astype(np.uint16)
                            elif col_max < 4294967295:
                                df[col] = df[col].astype(np.uint32)
                        else:  # signed
                            if col_min >= -128 and col_max <= 127:
                                df[col] = df[col].astype(np.int8)
                            elif col_min >= -32768 and col_max <= 32767:
                                df[col] = df[col].astype(np.int16)
                            elif col_min >= -2147483648 and col_max <= 2147483647:
                                df[col] = df[col].astype(np.int32)

                    elif pd.api.types.is_float_dtype(df[col]):
                        if abs(col_max) < 3.4e38 and abs(col_min) < 3.4e38:
                            df[col] = df[col].astype(np.float32)

                self.tables[table_name] = df
                new_size = df.memory_usage(deep=True).sum()

                if new_size < original_size:
                    saved_mb = (original_size - new_size) / 1024 / 1024
                    logger.info(f"Оптимизирована таблица {table_name}: сэкономлено {saved_mb:.2f} MB")

            except Exception as e:
                logger.warning(f"Не удалось оптимизировать таблицу {table_name}: {e}")
