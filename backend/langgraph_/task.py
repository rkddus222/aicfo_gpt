from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from backend.llm_models.model import llm
import psycopg2
import re

def load_prompt(prompt_path: str) -> str:
    """
    입력된 경로에 존재하는 프롬프트 파일을 로드합니다.

    Args:
        prompt_path (str): 프롬프트 파일의 경로.

    Returns:
        str: 로드된 프롬프트 내용.
    """
    with open(f"{prompt_path}", "r", encoding="utf-8") as f:
        prompt = f.read()

    return prompt

def extract_table_name_from_text(result: str) -> str:
    lines = result.split("\n")  # 줄바꿈으로 분리
    for line in lines:
        if line.startswith("- 조회 필요 테이블:"):  # 해당 키워드로 시작하는 줄 찾기
            # 콜론 뒤의 값을 가져온 후 괄호 및 내용 제거
            raw_value = line.split(":", 1)[1].strip()
            cleaned_value = re.sub(r"\s*\(.*?\)", "", raw_value)  # 괄호와 그 안의 내용 제거
            return cleaned_value
    return None  # 값이 없으면 None 반환


def evaluate_user_question(user_question: str) -> str:
    """사용자의 질문이 일상적인 대화문인지, 데이터 및 비즈니스와 관련된 질문인지를
    판단하는 역할을 담당하고 있는 함수입니다.
    현재는 LLM(gpt-4o-mini)을 통해 사용자의 질문을 판단하고 있습니다.

    Args:
        user_question (str): 사용자의 질문

    Returns:
        str: "1" : 데이터 또는 비즈니스와 관련된 질문, "0" : 일상적인 대화문
    """
    output_parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=load_prompt("backend/prompts/question_evaluation/main_v1.prompt")
            ),
            (
                "human",
                "질문(user_question): {user_question}",
            ),
        ]
    )
    chain = prompt | llm | output_parser

    output = chain.invoke({"user_question": user_question})
    return output


def simple_conversation(user_question: str) -> str:
    """사용자의 질문이 일상적인 대화문이라고 판단되었을 경우
    사용자와 일상적인 대화를 진행하는 함수입니다.
    현재는 LLM(gpt-4o-mini)으로 대응을 진행하고 있습니다.

    Args:
        user_question (str): 사용자의 일상적인 질문

    Returns:
        str: 사용자의 일상적인 질문에 대한 AI의 대답
    """
    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=load_prompt("backend/prompts/general_conversation/main_v1.prompt")
            ),
            (
                "human",
                "{user_question}",
            ),
        ]
    )
    chain = prompt | llm | output_parser

    output = chain.invoke({"user_question": user_question})
    return output


def analyze_user_question(user_question: str) -> str:
    output_parser = StrOutputParser()

    ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=load_prompt("backend/prompts/question_analysis/main_v1.prompt")
            ),
            (
                "human",
                "사용자 질문: {user_question}"
                + load_prompt("backend/prompts/question_analysis/human_v1.prompt"),
            ),
        ]
    )

    analyze_chain = ANALYZE_PROMPT | llm | output_parser
    analyze_question = analyze_chain.invoke({"user_question": user_question})
    return analyze_question

def create_query(
        user_question,
        user_question_analyze,
        selected_table,
        today,
        flow_status="KEEP",
):
    """
    user_question을 기반으로 prefix_v1 프롬프트를 활용하여 SQL 쿼리를 생성하는 함수.

    Args:
        user_question (str): 사용자의 질문.
        flow_status (str): 흐름 상태 ("KEEP"일 때만 동작).

    Returns:
        str: 생성된 SQL 쿼리.
    """
    try:
        if flow_status == "KEEP":
            # 프롬프트 로드 및 구성
            prefix_filename = f"backend/prompts/query_creation/{selected_table}.prompt"
            prefix = load_prompt(prefix_filename).format(
                user_question=user_question,
                user_question_analyze=user_question_analyze,
                today=today,
                context="",
            )

            # LLM 호출 및 출력 받기
            output_message = llm.invoke(prefix)  # LLM 응답 (AIMessage 객체)
            output = output_message.content  # 문자열로 변환

            # 출력에서 SQL 쿼리 추출
            match = re.search(r"```sql\s*(.*?)\s*```", output, re.DOTALL)
            if match:
                sql_query = match.group(1)
            else:
                match = re.search(r"SELECT.*?;", output, re.DOTALL)
                if match:
                    sql_query = match.group(0)
                else:
                    print("LLM Output 확인 필요:", output)
                    raise ValueError("SQL 쿼리를 찾을 수 없습니다.")

            return sql_query.strip()
        else:
            raise ValueError("flow_status가 'KEEP'이 아닙니다.")

    except Exception as e:
        print("\n=== 에러 발생 ===")
        print(f"에러 타입: {type(e)}")
        print(f"에러 메시지: {str(e)}")
        raise

def execute_query(host, database, user, password, port, query):
    """
    PostgreSQL 데이터베이스에 연결하고 쿼리를 실행하는 함수

    Args:
        host (str): 데이터베이스 호스트 (IP 또는 도메인)
        database (str): 데이터베이스 이름
        user (str): 사용자 이름
        password (str): 비밀번호
        port (int): 포트 번호
        query (str): 실행할 SQL 쿼리

    Returns:
        dict: {"columns": 열 이름 리스트, "rows": 데이터 리스트}
    """
    try:
        # 데이터베이스 연결
        connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )

        # 커서 생성
        cursor = connection.cursor()

        # SQL 쿼리 실행
        cursor.execute(query)

        # 결과 가져오기
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]  # 열 이름 추출

        # 연결 닫기
        cursor.close()
        connection.close()

        print("PostgreSQL 데이터베이스 연결 종료")
        return {"columns": columns, "rows": rows}

    except psycopg2.Error as e:
        print("PostgreSQL 연결 오류:")
        print("오류 코드:", e.pgcode)
        print("오류 메시지:", e.pgerror)
        print("전체 오류:", str(e))
        return None

def business_conversation(user_question, sql_query, query_result) -> str:
    output_parser = StrOutputParser()

    instruction = load_prompt("backend/prompts/sql_conversation/main_v1.prompt").format(
        sql_query=sql_query, query_result=query_result
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=instruction),
            (
                "human",
                """user_question: {user_question}""",
            ),
        ]
    )
    chain = prompt | llm | output_parser

    output = chain.invoke({"user_question": user_question})
    return output
