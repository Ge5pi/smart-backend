# services/dataframe_manager.py
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from sqlalchemy import inspect, text
import numpy as np

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
        self.total_memory_usage = 0

    def load_all_tables(self, max_rows_per_table: int = 100000) -> Dict[str, pd.DataFrame]:
        """Загружает все таблицы в память как DataFrame"""
        inspector = inspect(self.engine)
        tables_loaded = 0

        logger.info("Начинаем загрузку всех таблиц в DataFrame...")

        for table_name in inspector.get_table_names():
            if table_name == 'alembic_version':
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
                limit_clause = f" LIMIT {max_rows_per_table}" if row_count > max_rows_per_table else ""
                query = f"SELECT * FROM {table_name}{limit_clause}"

                # Загружаем данные
                df = pd.read_sql(query, self.engine)

                if not df.empty:
                    # Обрабатываем проблемные типы данных
                    df = self._clean_dataframe(df)

                    self.tables[table_name] = df

                    # Сохраняем схему таблицы
                    columns = inspector.get_columns(table_name)
                    primary_keys = inspector.get_pk_constraint(table_name).get('constrained_columns', [])

                    self.table_schemas[table_name] = {
                        'columns': {col['name']: str(col['type']) for col in columns},
                        'row_count': len(df),
                        'primary_keys': primary_keys,
                        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                    }

                    tables_loaded += 1
                    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                    self.total_memory_usage += memory_mb

                    logger.info(f"✅ Загружена таблица {table_name}: {len(df)} строк, {memory_mb:.2f} MB")

            except Exception as e:
                logger.error(f"❌ Ошибка загрузки {table_name}: {e}")

        # Анализируем связи между таблицами
        self._discover_relations(inspector)

        logger.info(f"🎉 Загружено {tables_loaded} таблиц, {len(self.relations)} связей, "
                    f"общий объем: {self.total_memory_usage:.2f} MB")

        return self.tables

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очищает DataFrame от проблемных значений"""
        # Заменяем inf на NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Обрабатываем проблемные типы данных
        for col in df.columns:
            if df[col].dtype == 'object':
                # Преобразуем None в строки для объектных колонок
                df[col] = df[col].astype(str).replace('None', '')
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Для числовых колонок оставляем NaN
                pass

        return df

    def _discover_relations(self, inspector):
        """Автоматически обнаруживает связи между таблицами"""
        for table_name in self.tables.keys():
            try:
                foreign_keys = inspector.get_foreign_keys(table_name)
                for fk in foreign_keys:
                    if fk['referred_table'] in self.tables:
                        relation = TableRelation(
                            from_table=table_name,
                            to_table=fk['referred_table'],
                            from_column=fk['constrained_columns'][0],
                            to_column=fk['referred_columns'][0],
                            relation_type='many_to_one'
                        )
                        self.relations.append(relation)

            except Exception as e:
                logger.warning(f"Не удалось получить FK для {table_name}: {e}")

        logger.info(f"Обнаружено {len(self.relations)} связей между таблицами")

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
                    common_col = list(common_cols)[0]
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

            logger.info(f"JOIN выполнен: {left_table} {join_type} {right_table}, "
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
                    # Автоопределение связи (упрощенная версия)
                    common_cols = set(result_df.columns).intersection(set(right_df.columns))
                    if common_cols:
                        common_col = list(common_cols)[0]
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
                except Exception as e:
                    logger.error(f"Ошибка фильтрации '{condition}': {e}")
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

        if result_df is None:
            result_df = pd.DataFrame()

        logger.info(f"Комплексный запрос завершен, результат: {len(result_df)} строк")
        return result_df

    def get_table_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по загруженным таблицам"""
        return {
            'total_tables': len(self.tables),
            'total_relations': len(self.relations),
            'total_memory_mb': self.total_memory_usage,
            'tables': {
                name: {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                }
                for name, df in self.tables.items()
            },
            'relations': [
                {
                    'from': rel.from_table,
                    'to': rel.to_table,
                    'on': f"{rel.from_column} -> {rel.to_column}"
                }
                for rel in self.relations
            ]
        }
