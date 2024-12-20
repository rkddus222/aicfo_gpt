from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# OpenAI API 키를 환경 변수에서 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")

# GPT-4 모델 설정
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    model_kwargs={"top_p": 0.9},  # top_p 값을 여기에 설정
    openai_api_key=OPENAI_API_KEY
)

# GPT-4 Turbo 모델 설정
closed_llm = ChatOpenAI(
    model="gpt-4", temperature=0, top_p=0, openai_api_key=OPENAI_API_KEY
)

# 동일 모델 다른 설정
solver_llm = ChatOpenAI(
    model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY
)
