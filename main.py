import io
import json
import os
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Optional, List
import numpy as np
import openai
import pandas as pd
import pinecone
from fastapi import APIRouter, Depends, FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from langdetect import detect, LangDetectException
from sqlalchemy.orm import Session
from starlette import status
from starlette.responses import Response, StreamingResponse
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from database_analytics import database_router
import auth
import config
import crud
import database
import models
import schemas
from config import API_KEY, PINECONE_KEY, redis_client, MESSAGE_LIMIT
from vk_auth import verify_vk_launch_params
from tg_router import telegram_mini_app_router
import services

api_key = API_KEY
pinecone_key = PINECONE_KEY

try:

    print("--- Successfully connected to Redis ---")

except Exception as e:
    print(f"FATAL: Could not connect to AWS or Redis on startup. Error: {e}")

LANG_MAP = {
    'ru': 'русском',
    'en': 'английском',
    'kz': 'казахском'
}

conf_mail = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)

client = openai.OpenAI(api_key=api_key)
pc = pinecone.Pinecone(api_key=pinecone_key)
app = FastAPI(title="SODA API")
app.include_router(database_router, tags=["Database Analytics"])


@app.on_event("startup")
async def startup_event():
    try:
        if redis_client:
            redis_client.ping()
            print("✅ Redis connected successfully")
        else:
            print("⚠️ Redis was not initialized")
    except Exception as e:
        print(f"❌ Redis ping failed: {e}")


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "ok"}


vk_mini_app_router = APIRouter(
    prefix="/vk-api",
    tags=["VK Mini App"]
)

allow_origin_regex = r"https://.*\.vk-apps\.com|https://.*\.vk\.com|https://web\.telegram\.org|https://smart-frontend-production\.up\.railway\.app|http://localhost:5173|https?://(www\.)?soda\.contact|https://soda-data\.netlify\.app"

EMBEDDING_MODEL = "text-embedding-3-small"
AGENT_MODEL = "gpt-4.1-nano"
CRITIC_MODEL = "gpt-4.1-nano"
INDEX_NAME = "soda-index"
BATCH_SIZE = 100

user_router = APIRouter()

origins = [
    "https://smart-frontend-production.up.railway.app",
    "http://localhost:5173",
    "https://soda.contact",
    "https://www.soda.contact",
    "https://soda-data.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    allow_origin_regex=allow_origin_regex
)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=1536, metric='cosine')
index = pc.Index(INDEX_NAME)

ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_df_from_s3(db: Session, file_id: str) -> pd.DataFrame:
    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File record not found in DB.")

    try:
        blob = config.gcs_bucket.blob(file_record.s3_path)
        file_content = blob.download_as_bytes()

        file_extension = Path(file_record.file_name).suffix.lower()
        if file_extension == '.csv':
            return pd.read_csv(io.StringIO(file_content.decode("utf-8")))
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(io.BytesIO(file_content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type in S3.")

    except Exception as e:
        print(f"Error reading file from S3: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read file from S3: {str(e)}")


def save_df_to_s3(db: Session, file_id: str, user_id: int, df: pd.DataFrame):
    """Безопасно сохраняет DataFrame в GCS, проверяя владельца."""
    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record or file_record.user_id != user_id:
        raise HTTPException(status_code=404, detail="File not found or access denied for saving.")
    try:
        with io.StringIO() as csv_buffer:
            df.to_csv(csv_buffer, index=False)
            blob = config.gcs_bucket.blob(file_record.s3_path)
            blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save processed file to GCS: {str(e)}")


@user_router.post("/users/register", response_model=schemas.User)
async def register_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Пользователь с таким email уже зарегистрирован")

    new_user = crud.create_user(db=db, user=user)

    code = crud.create_verification_code(db, new_user)

    message = MessageSchema(
        subject="Подтверждение регистрации SODA",
        recipients=[new_user.email],
        body=f"Ваш код подтверждения: {code}. Он действителен 15 минут.",
        subtype="plain"
    )

    fm = FastMail(conf_mail)
    await fm.send_message(message)

    return new_user


@user_router.post("/users/verify-email", summary="Подтверждение email по коду")
async def verify_email(verification_data: schemas.EmailVerification, db: Session = Depends(database.get_db)):
    user = crud.verify_email_code(db, verification_data.email, verification_data.code)
    if not user:
        raise HTTPException(status_code=400, detail="Неверный email или код подтверждения.")
    return {"message": "Email успешно подтвержден."}


@user_router.post("/users/password-reset/request", summary="Запрос на сброс пароля")
async def request_password_reset(request_data: schemas.PasswordResetRequest, db: Session = Depends(database.get_db)):
    user = crud.get_user_by_email(db, request_data.email)
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден.")

    token = crud.create_password_reset_token(db, user)
    reset_link = f"http://www.soda.contact/reset-password?token={token}"  # Замените на реальный URL

    message = MessageSchema(
        subject="Сброс пароля SODA",
        recipients=[user.email],
        body=f"Для сброса пароля перейдите по ссылке: {reset_link}",
        subtype="plain"
    )

    fm = FastMail(conf_mail)
    await fm.send_message(message)

    return {"message": "Инструкции по сбросу пароля отправлены на ваш email."}


@user_router.post("/users/password-reset", summary="Подтверждение сброса пароля")
async def confirm_password_reset(
        reset_data: schemas.PasswordReset,
        db: Session = Depends(database.get_db)
):
    """
    Завершает процесс сброса пароля, используя токен и новый пароль.
    """
    success = crud.reset_user_password(
        db, token=reset_data.token, new_password=reset_data.new_password
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Неверный, просроченный или уже использованный токен."
        )
    return {"message": "Пароль успешно обновлен."}


@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = auth.authenticate_user(db, email=form_data.username, password=form_data.password)
    if not user:
        db_user = crud.get_user_by_email(db, form_data.username)
        if db_user and not db_user.is_verified:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Email не подтвержден. Проверьте почту.",
                                headers={"WWW-Authenticate": "Bearer"})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверный email или пароль",
                            headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@user_router.get("/files/me", response_model=list[schemas.File])
def read_user_files(db: Session = Depends(database.get_db),
                    current_user: models.User = Depends(auth.get_current_active_user)):
    return crud.get_files_by_user_id(db=db, user_id=current_user.id)


@user_router.get("/users/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(auth.get_current_active_user)):
    return current_user


@user_router.post("/users/subscribe", response_model=schemas.SubscriptionOrder, tags=["Users"])
async def create_subscription(
        order_data: schemas.SubscriptionOrderCreate,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Принимает заявку на подписку от пользователя."""

    # 1. Создаем заявку в БД
    order = crud.create_subscription_order(db=db, order=order_data, user_id=current_user.id)

    # 2. Готовим и отправляем письмо администратору (вам)
    admin_message = MessageSchema(
        subject=f"Новая заявка на подписку от {current_user.email}",
        recipients=["danyagespi@gmail.com"],
        body=f"""
        Поступила новая заявка на подписку:
        - Имя клиента: {order.customer_name}
        - Email пользователя: {current_user.email}
        - Время заявки: {order.created_at.strftime('%Y-%m-%d %H:%M:%S')} UTC
        """,
        subtype="plain"
    )

    # 3. Готовим и отправляем письмо-подтверждение пользователю
    user_message = MessageSchema(
        subject="Ваша заявка на подписку SODA принята",
        recipients=[current_user.email],
        body=f"""
        Здравствуйте, {order.customer_name}!

        Спасибо за вашу заявку на подписку. Мы получили ее и свяжемся с вами в течение часа для уточнения деталей.

        С уважением,
        Команда SODA
        """,
        subtype="plain"
    )

    fm = FastMail(conf_mail)
    await fm.send_message(admin_message)
    await fm.send_message(user_message)

    return order


app.include_router(user_router, tags=["Users"])


@app.get("/files/{file_id}/sessions", response_model=List[schemas.ChatSessionInfo], tags=["AI Agent"])
async def get_file_sessions(
        file_id: str,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Возвращает список всех сессий чата для указанного файла."""
    sessions = crud.get_sessions_for_file(db, user_id=current_user.id, file_uid=file_id)
    return sessions


@app.post("/sessions/create", response_model=schemas.ChatSessionInfo, tags=["AI Agent"])
async def create_new_session(
        data: schemas.SessionCreate,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Создает новую сессию чата для файла."""
    session = crud.create_chat_session(db, user_id=current_user.id, file_uid=data.file_id)
    return session


@app.get("/sessions/{session_id}/history", response_model=schemas.HistoryResponse, tags=["AI Agent"])
async def get_session_history(
        session_id: str,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    """Получает историю сообщений для конкретной сессии."""
    session = crud.get_session_by_id(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Сессия не найдена или у вас нет к ней доступа.")

    messages = crud.get_chat_messages(db, session_id)
    return {"history": messages}


def format_row(row_index: int, row: pd.Series) -> str:
    return f"Row-{row_index + 1}: " + " | ".join([f"{col}: {row[col]}" for col in row.index])


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(6))
def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [r.embedding for r in response.data]


def impute_with_sklearn(df: pd.DataFrame, selected_columns: list) -> tuple[pd.DataFrame, dict]:
    df_imputed = df.copy()
    processing_results = {}

    numeric_cols_to_impute = [
        col for col in selected_columns if col in df_imputed.columns and pd.api.types.is_numeric_dtype(df_imputed[col])
    ]
    categorical_cols_to_impute = [
        col for col in selected_columns if
        col in df_imputed.columns and not pd.api.types.is_numeric_dtype(df_imputed[col])
    ]

    if numeric_cols_to_impute:
        print(f"Применение KNNImputer к столбцам: {numeric_cols_to_impute}")
        numeric_df = df_imputed.select_dtypes(include=np.number)

        knn_imputer = KNNImputer(n_neighbors=5)

        imputed_data = knn_imputer.fit_transform(numeric_df)

        imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns, index=numeric_df.index)

        for col in numeric_cols_to_impute:
            df_imputed[col] = imputed_df[col]
            processing_results[col] = "Пропуски заполнены методом k-ближайших соседей (KNN)."

    if categorical_cols_to_impute:
        print(f"Применение SimpleImputer к столбцам: {categorical_cols_to_impute}")
        simple_imputer = SimpleImputer(strategy='most_frequent')

        df_imputed[categorical_cols_to_impute] = simple_imputer.fit_transform(
            df_imputed[categorical_cols_to_impute]
        )
        for col in categorical_cols_to_impute:
            processing_results[col] = "Пропуски заполнены наиболее частым значением."

    return df_imputed, processing_results


def get_critic_evaluation(query: str, answer: str) -> dict:
    critic_prompt = f""" You are a meticulous AI data analyst critic. Your task is to evaluate a generated answer 
    based on a user's query. Provide your evaluation in a structured JSON format. 

        **User Query:**
        {query}

        **Answer to Evaluate:**
        {answer}

        **Evaluation Criteria:** 1.  **relevance**: Is the answer directly related to the user's query? (1-5, 
        5 is most relevant) 2.  **completeness**: Does the answer fully address all parts of the query? (1-5, 
        5 is most complete) 3.  **accuracy**: How likely is the answer to be factually correct in a pandas DataFrame 
        context? Does it seem plausible? (1-5, 5 is most accurate) 4.  **feedback**: Provide CONCISE and ACTIONABLE 
        feedback. - If the answer is good, write "The answer is sufficient.". - If the answer is bad, explain WHY it 
        is bad. For example: "The answer hallucinates information not present in the query" or "The calculation seems 
        incorrect for the requested metric." 5.  **suggestion**: If the answer is poor, suggest a better approach. 
        For example: "It would be better to use `df.groupby('category')['sales'].sum()`" or "A better approach would 
        be to use the RAG tool to find the specific row. Nevertheless, the answer must be as short and exact as 
        possible unless it wasn't asked to explain something" 

        **Output (JSON format only):**
        """
    try:
        response = client.chat.completions.create(
            model=CRITIC_MODEL,
            messages=[{"role": "user", "content": critic_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in critic model: {e}")
        return {"relevance": 5, "completeness": 5, "confidence": 5,
                "feedback": "Critic model failed, assuming success."}


def get_refined_answer(history: list, original_answer: str, feedback: str, suggestion: str, lang_name: str) -> str:
    refiner_prompt = f"""Ты — эксперт по анализу данных. Твоя предыдущая попытка ответить на вопрос пользователя была 
    неудачной. Критик предоставил отзыв. Твоя задача — сгенерировать новый, финальный и правильный ответ, 
    который учитывает этот отзыв. 

        **Исходная история диалога:**
        {json.dumps(history, indent=2, ensure_ascii=False)}

        **Твой предыдущий (неудовлетворительный) ответ:**
        {original_answer}

        **Отзыв критика (что было не так):**
        {feedback}

        **Предложение критика (как это исправить):**
        {suggestion}

        **Твоя задача:** Сгенерируй новый, улучшенный и полный ответ **на {lang_name} языке**. Новый ответ должен 
        напрямую отвечать на последний вопрос пользователя и исправлять проблемы, поднятые критиком. Не повторяй 
        ошибок. Представь финальный результат пользователю в ясной и понятной форме. """
    response = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[{"role": "user", "content": refiner_prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content


SYSTEM_PROMPT = """Ты — элитный AI-аналитик данных SODA. Твоя работа — помогать пользователю анализировать данные в 
pandas DataFrame `df`. 

**Ключевые правила:**

1.  **Состояние DataFrame:** Объект `df` существует только в рамках сессии. Изменения, внесенные через 
`execute_python_code`, сохраняются между запросами, но **не в исходном файле**. Всегда работай с актуальным 
состоянием `df`. 

2.  **Думай шаг за шагом:** Прежде чем выбрать инструмент, четко определи **все** цели пользователя в его запросе.

3.  **Многозадачность:** Пользователь может дать несколько команд в одном запросе (например, 'удали дубликаты и 
сохрани файл'). Внимательно проанализируй запрос, определи все необходимые действия и спланируй вызов **нескольких 
инструментов подряд** в правильном порядке. Например, сначала вызови `execute_python_code` для модификации `df`, 
а затем `save_dataframe_to_file` для сохранения. 

4.  **Важность сохранения:** Изменения в `df` не сохраняются в файл автоматически. После **каждой успешной 
модификации DataFrame** с помощью `execute_python_code`, вежливо напомни пользователю об этом. * **Пример твоего 
напоминания:** "Готово! Я изменил данные. Напоминаю, что изменения сохранены только в рамках нашей сессии. Чтобы 
записать их в файл, скажите 'сохрани результат'." 

5.  **Выбор инструмента:** * `execute_python_code`: Для любых вычислений, фильтрации, агрегации, модификации `df` или 
получения мета-информации (`df.describe()`, `df.shape`). * `answer_question_from_context`: Только для смыслового 
поиска по текстовому содержанию строк. * `save_dataframe_to_file`: Только по явной просьбе пользователя "сохрани", 
"запиши" и т.д. 

6.  **Обработка ошибок:** Если твой код вызывает ошибку, не извиняйся. Проанализируй ошибку, исправь код в следующем 
шаге и попробуй снова. 

7.  **Язык:** Всегда отвечай на том же языке, на котором задан вопрос.
"""


def execute_python_code(df: pd.DataFrame, code: str) -> tuple[pd.DataFrame, str]:
    print(f"TOOL: Выполнение кода:\n---\n{code}\n---")
    local_scope = {"df": df.copy(), "pd": pd}
    try:
        lines = code.strip().split('\n')
        if len(lines) == 1:
            output = eval(code, {"pd": pd}, local_scope)
        else:
            exec_lines, eval_line = lines[:-1], lines[-1]
            exec("\n".join(exec_lines), {"pd": pd}, local_scope)
            output = eval(eval_line, {"pd": pd}, local_scope)

        final_df = local_scope['df']

        if isinstance(output, pd.DataFrame):
            return final_df, output.head(15).to_markdown()
        if isinstance(output, pd.Series):
            return final_df, output.to_frame().head(15).to_markdown()
        return final_df, str(output)
    except Exception:
        try:
            exec(code, {"pd": pd}, local_scope)
            final_df = local_scope['df']
            return final_df, "Код выполнен, DataFrame обновлен."
        except Exception as exec_e:
            return df, f"Ошибка выполнения кода: {exec_e}"


def run_rag_pipeline(file_id: str, query: str, lang_name: str) -> str:
    query_embedding = get_embeddings([query])[0]
    search_results = index.query(vector=query_embedding, top_k=7, filter={"file_id": file_id}, include_metadata=True)
    context = " "
    if search_results.get('matches'):
        for match in search_results['matches']:
            context += match['metadata']['original_text'] + "\n---\n"
    if not context.strip():
        return "Не удалось найти релевантную информацию в файле."

    rag_system_prompt = f"Ответь на вопрос пользователя **на {lang_name} языке**, основываясь ИСКЛЮЧИТЕЛЬНО на " \
                        f"предоставленном контексте. "
    rag_messages = [
        {"role": "system", "content": rag_system_prompt},
        {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {query}"}
    ]
    response = client.chat.completions.create(model=AGENT_MODEL, messages=rag_messages, temperature=0.0)
    return response.choices[0].message.content


def save_dataframe_to_file(db: Session, file_id: str, df_to_save: pd.DataFrame) -> str:
    print(f"TOOL: Сохранение файла {file_id}...")

    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record:
        return "Ошибка: Запись о файле не найдена в БД."

    try:
        with io.StringIO() as csv_buffer:
            df_to_save.to_csv(csv_buffer, index=False)
            blob = config.gcs_bucket.blob(file_record.s3_path)
            blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
        return f"Файл '{file_record.file_name}' успешно сохранен."
    except Exception as e:
        return f"Ошибка при сохранении файла в S3: {e}"


def answer_question_from_context(file_id: str, query: str, lang_name: str) -> str:
    print(f"TOOL: RAG-запрос для файла {file_id} на языке '{lang_name}': '{query}'")
    return run_rag_pipeline(file_id, query, lang_name)


tools_definition = [
    {"type": "function", "function": {
        "name": "execute_python_code",
        "description": "Выполняет Python-код для манипуляций с DataFrame `df`. Используется для расчетов, агрегации, "
                       "фильтрации, модификации данных.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Код на Python для выполнения над DataFrame 'df'."}
            },
            "required": ["code"]
        }
    }},
    {"type": "function", "function": {
        "name": "answer_question_from_context",
        "description": "Ищет в файле строки по смысловому содержанию и отвечает на вопрос на их основе. Пример: 'Что "
                       "можешь рассказать о заказе A-123?', 'Опиши клиента с самой большой суммой покупки'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Вопрос для поиска по смыслу в документе."}
            },
            "required": ["query"]
        }
    }},
    {"type": "function", "function": {
        "name": "save_dataframe_to_file",
        "description": "Сохраняет текущий DataFrame `df` в исходный файл. Использовать только по прямой и явной "
                       "просьбе пользователя, например: 'Сохрани результат'.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }},
]
available_functions = {
    "execute_python_code": execute_python_code,
    "answer_question_from_context": answer_question_from_context,
    "save_dataframe_to_file": save_dataframe_to_file,
}


@app.post("/upload/", tags=["File Operations"])
async def upload_file(
        file: UploadFile = File(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(auth.get_current_active_user)
):
    file_id = str(uuid.uuid4())
    original_filename = file.filename
    file_extension = Path(original_filename).suffix.lower()

    if file_extension not in ['.csv', '.xlsx', '.xls']:
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип файла.")

    s3_path = f"uploads/{current_user.id}/{file_id}/{original_filename}"

    try:
        contents = await file.read()
        blob = config.gcs_bucket.blob(s3_path)
        blob.upload_from_string(contents, content_type=file.content_type)

        db_file = crud.create_user_file(db=db, user_id=current_user.id, file_uid=file_id, file_name=original_filename,
                                        s3_path=s3_path)

        file_stream = io.BytesIO(contents)
        if file_extension == '.csv':
            df = pd.read_csv(file_stream, encoding="utf-8")
        else:
            df = pd.read_excel(file_stream)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке или чтении файла: {e}")

    df = df.replace([np.inf, -np.inf], np.nan)

    vectors_to_upsert = []
    df_rag = df.where(pd.notnull(df), 'null')
    for i in tqdm(range(len(df_rag)), desc=f"Подготовка векторов для {file_id}"):
        row_text = format_row(i, df_rag.iloc[i])
        vectors_to_upsert.append({
            "id": f"{file_id}-row-{i}", "values": [],
            "metadata": {"file_id": file_id, "original_text": row_text}
        })

    for i in tqdm(range(0, len(vectors_to_upsert), BATCH_SIZE), desc=f"Индексация в Pinecone для {file_id}"):
        batch = vectors_to_upsert[i:i + BATCH_SIZE]
        texts_to_embed = [v['metadata']['original_text'] for v in batch]
        try:
            embeddings = get_embeddings(texts_to_embed)
            for j, vector in enumerate(batch):
                vector["values"] = embeddings[j]
            index.upsert(vectors=batch)
        except Exception as e:
            print(f"Ошибка при обработке пачки {i}-{i + BATCH_SIZE}: {e}")

    analysis = [{"column": col, "dtype": str(df[col].dtype), "nulls": int(df[col].isna().sum()),
                 "unique": int(df[col].nunique())} for col in df.columns]
    preview_data = df.fillna("null").to_dict(orient="records")
    return {"columns": analysis, "preview": preview_data, "file_id": file_id, "file_name": original_filename}


@app.post("/sessions/ask", tags=["AI Agent"])
async def session_ask(session_id: str = Form(...), query: str = Form(...), db: Session = Depends(database.get_db),
                      current_user: models.User = Depends(auth.get_current_active_user)):
    if (not current_user.is_active) and current_user.messages_used >= MESSAGE_LIMIT:
        raise HTTPException(
            status_code=403,
            detail="Вы использовали все бесплатные сообщения. Пожалуйста, перейдите на платный тариф."
        )

    # Проверяем наличие сессии в БД
    session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id,
                                                  models.ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Сессия не найдена или у вас нет к ней доступа.")

    # Сохраняем сообщение пользователя в БД
    crud.add_chat_message(db, session_id=session_id, role="user", content=query)

    # Загружаем историю из БД для контекста
    db_messages = crud.get_chat_messages(db, session_id=session.id)
    messages = [{"role": msg.role, "content": msg.content} for msg in db_messages]

    try:
        lang_code = detect(query)
    except LangDetectException:
        lang_code = 'ru'
    lang_name = LANG_MAP.get(lang_code, lang_code)

    df = get_df_from_s3(db, session.file.file_uid)

    buf = io.StringIO()
    df.info(buf=buf, verbose=False)
    df_info = buf.getvalue()
    df_head = df.head().to_markdown()

    # Вставляем системный промпт в начало (не сохраняем его в БД)
    system_prompt_message = {
        "role": "system",
        "content": SYSTEM_PROMPT + f"\n\nВАЖНО: Текущий язык общения - {lang_name}. Все ответы должны быть на этом языке."
    }

    contextual_query = f"Контекст данных:\n1. Схема данных (df.info()):\n```\n{df_info}```\n2. Первые 5 строк (df.head()):\n```\n{df_head}```\n---\nВопрос пользователя: {query}"

    # Заменяем последний user-вопрос на расширенный с контекстом
    messages[-1]['content'] = contextual_query
    messages.insert(0, system_prompt_message)

    try:
        final_answer = ""
        for _ in range(5):
            response = client.chat.completions.create(model=AGENT_MODEL, messages=messages, tools=tools_definition,
                                                      tool_choice="auto")
            response_message = response.choices[0].message
            # Важно: `messages` обновляется внутри цикла для сохранения контекста вызова инструментов
            messages.append(response_message.model_dump(exclude_unset=True))

            if not response_message.tool_calls:
                final_answer = response_message.content
                break

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Добавляем file_id в аргументы для функций, которым он нужен
                if function_name in ["answer_question_from_context", "save_dataframe_to_file"]:
                    function_args["file_id"] = session.file.file_uid

                if function_name == "execute_python_code":
                    df, function_response = execute_python_code(df=df, code=function_args.get("code"))
                elif function_name == "save_dataframe_to_file":
                    function_response = save_dataframe_to_file(db=db, file_id=function_args.get("file_id"),
                                                               df_to_save=df)
                elif function_name == "answer_question_from_context":
                    function_response = answer_question_from_context(file_id=function_args.get("file_id"),
                                                                     query=function_args.get("query"),
                                                                     lang_name=lang_name)
                else:
                    function_response = f"Ошибка: неизвестный инструмент {function_name}"

                messages.append(
                    {"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response})

        if not final_answer:
            # Убираем системный промпт перед финальным запросом, если он не нужен
            final_messages = [msg for msg in messages if msg.get("role") != "system"]
            last_response = client.chat.completions.create(model=AGENT_MODEL, messages=final_messages)
            final_answer = last_response.choices[0].message.content

        evaluation = get_critic_evaluation(query, final_answer)
        if evaluation.get("accuracy", 5) < 4:
            # Передаем историю без системного промпта
            history_for_refiner = [msg for msg in messages if msg.get("role") != "system"]
            refined_answer = get_refined_answer(
                history=history_for_refiner,
                original_answer=final_answer,
                feedback=evaluation.get("feedback"),
                suggestion=evaluation.get("suggestion", ""),
                lang_name=lang_name
            )
            final_answer = refined_answer

        # Сохраняем ответ ассистента в БД
        if final_answer:
            crud.add_chat_message(db, session_id=session_id, role="assistant", content=final_answer)

        crud.increment_usage_counter(db, user=current_user, counter_type='messages')

        # Удаляем логику Redis для хранения сообщений
        return {"answer": final_answer, "evaluation": evaluation}
    except Exception as e:
        print(f"Error in session_ask (session: {session_id}): {e}")
        raise HTTPException(status_code=500, detail=f"Произошла внутренняя ошибка: {str(e)}")


@app.post("/analyze/", tags=["File Operations"])
async def analyze_csv(file_id: str = Form(...), db: Session = Depends(database.get_db)):
    df = get_df_from_s3(db, file_id)

    analysis = [{"column": col, "dtype": str(df[col].dtype), "nulls": int(df[col].isna().sum()),
                 "unique": int(df[col].nunique()), "sample_values": df[col].dropna().astype(str).unique()[:3].tolist()}
                for col in df.columns]
    return {"columns": analysis}


@app.post("/impute-missing/", tags=["Data Cleaning"])
async def impute_missing_values_endpoint(file_id: str = Form(...), columns: Optional[str] = Form(None),
                                         db: Session = Depends(database.get_db)):
    if not columns:
        raise HTTPException(status_code=400, detail="Столбцы для импутации не выбраны.")

    df = get_df_from_s3(db, file_id)
    selected_columns = json.loads(columns)

    if not isinstance(selected_columns, list) or not selected_columns:
        raise HTTPException(status_code=400, detail="Неверный формат списка столбцов.")

    missing_before = {
        col: int(df[col].isna().sum()) for col in selected_columns if col in df.columns
    }

    df_imputed, processing_results = impute_with_sklearn(df, selected_columns)

    missing_after = {
        col: int(df_imputed[col].isna().sum()) for col in selected_columns if col in df_imputed.columns
    }

    for col in selected_columns:
        if col not in df.columns:
            processing_results[col] = "Ошибка: Столбец не найден в файле."
            missing_before[col] = "N/A"
            missing_after[col] = "N/A"

    try:
        file_record = crud.get_file_by_uid(db, file_id)
        with io.StringIO() as csv_buffer:
            df_imputed.to_csv(csv_buffer, index=False)
            blob = config.gcs_bucket.blob(file_record.s3_path)
            blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось сохранить обработанный файл в GCS: {str(e)}")

    return {
        "processing_results": processing_results,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "preview": df_imputed.fillna("null").to_dict(orient="records"),
    }


@app.post("/outliers/", tags=["Data Cleaning"])
async def detect_outliers_endpoint(file_id: str = Form(...), columns: Optional[str] = Form(None),
                                   db: Session = Depends(database.get_db)):
    df = get_df_from_s3(db, file_id)

    if columns:
        selected_columns = json.loads(columns)
    else:
        selected_columns = df.select_dtypes(include=np.number).columns.tolist()

    if not selected_columns:
        return {"outlier_count": 0, "outlier_preview": []}

    valid_columns = [col for col in selected_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not valid_columns:
        return {"outlier_count": 0, "outlier_preview": []}

    numeric_df = df[valid_columns].dropna()
    if numeric_df.empty:
        return {"outlier_count": 0, "outlier_preview": []}

    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(numeric_df)
    outlier_indices = numeric_df.index[predictions == -1]
    outliers_df = df.loc[outlier_indices]
    outliers_preview = outliers_df.fillna("null").to_dict('records')
    return {"outlier_count": len(outliers_df), "outlier_preview": outliers_preview}


@app.get("/download-cleaned/{file_id}", tags=["File Operations"])
async def download_cleaned_file(file_id: str, db: Session = Depends(database.get_db)):
    """Отдает пользователю последнюю версию файла напрямую из S3."""
    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File record not found in DB.")

    try:
        blob = config.gcs_bucket.blob(file_record.s3_path)
        content = blob.download_as_bytes()
        return Response(
            content=content,
            media_type='text/csv',
            headers={"Content-Disposition": f'attachment; filename=\"{file_record.file_name}\"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch file from S3: {e}")


@app.post("/analyze-existing/", tags=["File Operations"])
async def analyze_existing_file(file_id: str = Form(...), db: Session = Depends(database.get_db)):
    df = get_df_from_s3(db, file_id)
    analysis = [{"column": col, "dtype": str(df[col].dtype), "nulls": int(df[col].isna().sum()),
                 "unique": int(df[col].nunique())} for col in df.columns]
    preview_data = df.fillna("null").to_dict(orient="records")
    return {"columns": analysis, "preview": preview_data}


@app.post("/chart-data/")
async def get_chart_data(
        file_id: str = Form(...),
        chart_type: str = Form(...),
        column1: str = Form(...),
        column2: Optional[str] = Form(None),
        column3: Optional[str] = Form(None),
        db: Session = Depends(database.get_db)
):
    df = get_df_from_s3(db, file_id)

    chart_data = {}

    try:
        if chart_type in ["histogram", "pie"]:
            if df[column1].dtype == 'object':
                counts = df[column1].dropna().value_counts().nlargest(15)
            else:
                counts = df[column1].dropna().value_counts()
            chart_data = {"labels": counts.index.astype(str).tolist(), "values": counts.values.tolist()}

        elif chart_type in ["line", "area"] and column2:
            try:
                df[column1] = pd.to_datetime(df[column1])
                line_df = df[[column1, column2]].dropna().sort_values(by=column1)
                chart_data = {
                    "labels": line_df[column1].dt.strftime('%Y-%m-%d').tolist(),
                    "values": line_df[column2].tolist(),
                }
            except Exception:
                raise HTTPException(status_code=400, detail=f"Не удалось преобразовать столбец '{column1}' в дату.")

        elif chart_type == "scatter" and column2:
            scatter_df = df[[column1, column2]].dropna()
            chart_data = {"points": scatter_df.to_dict(orient='records')}

        elif chart_type == "bubble" and column2 and column3:
            bubble_df = df[[column1, column2, column3]].dropna()
            for col in [column1, column2, column3]:
                if not pd.api.types.is_numeric_dtype(bubble_df[col]):
                    raise HTTPException(status_code=400,
                                        detail=f"Для пузырьковой диаграммы все столбцы должны быть числовыми. Столбец '{col}' не является числовым.")

            chart_data = {"points": bubble_df.to_dict(orient='records')}

        elif chart_type == "boxplot":
            if not pd.api.types.is_numeric_dtype(df[column1]):
                raise HTTPException(status_code=400,
                                    detail=f"Для 'Ящика с усами' столбец '{column1}' должен быть числовым.")

            stats = df[column1].describe()
            q1 = stats['25%']
            q3 = stats['75%']
            iqr = q3 - q1
            lower_whisker = q1 - 1.5 * iqr
            upper_whisker = q3 + 1.5 * iqr

            chart_data = {
                "min": stats['min'], "q1": q1, "median": stats['50%'],
                "q3": q3, "max": stats['max'],
                "outliers": df[(df[column1] < lower_whisker) | (df[column1] > upper_whisker)][column1].tolist()
            }

        else:
            raise HTTPException(status_code=400, detail="Неверный тип графика или нехватка данных.")

        return {"chart_type": chart_type, "data": chart_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при подготовке данных для графика: {str(e)}")


@app.post("/encode-categorical/", tags=["Data Cleaning"])
async def encode_categorical_features(
        file_id: str = Form(...),
        columns: str = Form(...),  # Ожидаем JSON-строку с массивом столбцов
        db: Session = Depends(database.get_db)
):
    df = get_df_from_s3(db, file_id)
    selected_columns = json.loads(columns)

    if not all(col in df.columns for col in selected_columns):
        raise HTTPException(status_code=404, detail="Один или несколько выбранных столбцов не найдены в файле.")

    for col in selected_columns:
        if df[col].nunique() > 50:
            raise HTTPException(
                status_code=400,
                detail=f"Столбец '{col}' имеет слишком много ({df[col].nunique()}) уникальных значений для кодирования. Максимум: 50."
            )

    try:
        df_encoded = pd.get_dummies(df, columns=selected_columns, prefix=selected_columns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при кодировании: {e}")

    try:
        file_record = crud.get_file_by_uid(db, file_id)
        with io.StringIO() as csv_buffer:
            df_encoded.to_csv(csv_buffer, index=False)
            blob = config.gcs_bucket.blob(file_record.s3_path)
            blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Не удалось сохранить закодированный файл в GCS: {str(e)}")

    new_analysis = [{"column": col, "dtype": str(df_encoded[col].dtype), "nulls": int(df_encoded[col].isna().sum()),
                     "unique": int(df_encoded[col].nunique())} for col in df_encoded.columns]
    new_preview = df_encoded.fillna("null").to_dict(orient="records")

    return {
        "message": "Категориальные столбцы успешно закодированы.",
        "columns": new_analysis,
        "preview": new_preview
    }


@app.get("/chart/{report_id}/{filename}")
def get_chart(report_id: int, filename: str):
    s3_key = f"charts/{report_id}/{filename}"
    blob = config.gcs_bucket.blob(s3_key)
    content = blob.download_as_bytes()

    # Определяем MIME-тип
    if filename.lower().endswith(".png"):
        media_type = "image/png"
    elif filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        media_type = "image/jpeg"
    elif filename.lower().endswith(".svg"):
        media_type = "image/svg+xml"
    else:
        media_type = "application/octet-stream"

    return StreamingResponse(io.BytesIO(content), media_type=media_type)


@app.get("/preview/{file_id}", tags=["File Operations"])
async def get_paginated_preview(
        file_id: str,
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
        db: Session = Depends(database.get_db)
):
    df = get_df_from_s3(db, file_id)

    start_index = (page - 1) * page_size
    end_index = start_index + page_size

    paginated_preview_df = df.iloc[start_index:end_index]

    return {
        "preview": paginated_preview_df.fillna("null").to_dict(orient="records"),
        "total_rows": len(df)
    }


@vk_mini_app_router.post("/upload/", response_model=schemas.File)
async def vk_upload_file(
        file: UploadFile = File(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(verify_vk_launch_params)
):
    return await services.logic_upload_file(file, db, current_user)


@vk_mini_app_router.get("/files/me", response_model=List[schemas.File])
def vk_read_user_files(
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(verify_vk_launch_params)
):
    return services.logic_read_user_files(db, current_user)


@vk_mini_app_router.post("/analyze-existing/")
async def vk_analyze_existing_file(
        file_id: str = Form(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(verify_vk_launch_params)
):
    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record or file_record.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ к файлу запрещен.")
    return services.logic_analyze_existing_file(file_id, db)


@vk_mini_app_router.post("/impute-missing/")
async def vk_impute_missing(
        file_id: str = Form(...),
        columns: str = Form(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(verify_vk_launch_params)
):
    return services.logic_impute_missing(file_id, columns, db, current_user)


@vk_mini_app_router.post("/outliers/")
async def vk_detect_outliers(
        file_id: str = Form(...),
        columns: str = Form(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(verify_vk_launch_params)
):
    return services.logic_detect_outliers(file_id, columns, db)


@vk_mini_app_router.post("/encode-categorical/")
async def vk_encode_categorical(
        file_id: str = Form(...),
        columns: str = Form(...),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(verify_vk_launch_params)
):
    return services.logic_encode_categorical(file_id, columns, db, current_user)


@vk_mini_app_router.get("/download-cleaned/{file_id}")
async def vk_download_cleaned_file(
        file_id: str,
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(verify_vk_launch_params)
):
    return services.logic_download_cleaned_file(file_id, db, current_user)


@vk_mini_app_router.get("/preview/{file_id}")
async def vk_get_paginated_preview(
        file_id: str,
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=200),
        db: Session = Depends(database.get_db),
        current_user: models.User = Depends(verify_vk_launch_params)
):
    file_record = crud.get_file_by_uid(db, file_id)
    if not file_record or file_record.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ к файлу запрещен.")
    return services.logic_get_paginated_preview(file_id, page, page_size, db)

app.include_router(vk_mini_app_router)
app.include_router(telegram_mini_app_router)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
