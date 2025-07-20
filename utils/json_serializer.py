# utils/json_serializer.py
import json
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Any
import logging

logger = logging.getLogger(__name__)


def convert_to_serializable(obj: Any) -> Any:
    """Конвертирует pandas/numpy объекты в JSON-serializable формат"""

    try:
        if obj is None or pd.isna(obj):
            return None
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return [convert_to_serializable(item) for item in obj.tolist()]
        elif isinstance(obj, pd.DataFrame):
            return convert_to_serializable(obj.to_dict('records'))
        elif isinstance(obj, pd.Series):
            return convert_to_serializable(obj.to_dict())
        elif isinstance(obj, dict):
            return {str(key): convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # datetime-like objects
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):  # Custom objects
            return convert_to_serializable(obj.__dict__)
        else:
            return obj
    except Exception as e:
        logger.warning(f"Не удалось конвертировать объект {type(obj)}: {e}")
        return str(obj)


def safe_json_serialize(data: Any) -> str:
    """Безопасная JSON сериализация с обработкой pandas объектов"""
    try:
        serializable_data = convert_to_serializable(data)
        return json.dumps(serializable_data, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка JSON сериализации: {e}")
        # Fallback к базовой сериализации
        return json.dumps({"error": f"Serialization failed: {str(e)}"}, ensure_ascii=False)


class DataFrameJSONEncoder(json.JSONEncoder):
    """JSON encoder с поддержкой pandas объектов"""

    def default(self, obj):
        try:
            return convert_to_serializable(obj)
        except Exception as e:
            logger.warning(f"JSONEncoder fallback для {type(obj)}: {e}")
            return super().default(obj)


def clean_dataframe_for_json(df: pd.DataFrame) -> list:
    """Специальная очистка DataFrame для JSON с максимальной совместимостью"""
    if df.empty:
        return []

    try:
        # Конвертируем все колонки в совместимые типы
        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].dtype == 'datetime64[ns]':
                df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
            elif df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).fillna('')
            elif pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(0)

        return convert_to_serializable(df_clean.to_dict('records'))

    except Exception as e:
        logger.error(f"Ошибка очистки DataFrame: {e}")
        return []
