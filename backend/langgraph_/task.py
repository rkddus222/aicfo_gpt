from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from backend.llm_models.model import llm
from typing import List
import psycopg2
from .utils import load_prompt
import re


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
                + load_prompt("LLMTEST/backend/prompts/question_analysis/human_v1.prompt"),
            ),
        ]
    )

    analyze_chain = ANALYZE_PROMPT | llm | output_parser
    analyze_question = analyze_chain.invoke({"user_question": user_question})

    return analyze_question


def check_leading_question(leading_question: str) -> int:
    if leading_question.startswith("종료") or leading_question.startswith('"종료'):
        return 0
    else:
        return 1


def refine_user_question(user_question: str, user_question_analyze: str) -> str:
    output_parser = StrOutputParser()
    REFINE_PROMPT = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=load_prompt("backend/prompts/question_refinement/main_v1.prompt")
            ),
            (
                "human",
                """사용자 질문: 
                {user_question}

                사용자 질문 분석: 
                {user_question_analyze}

                구체화된 질문:""",
            ),
        ]
    )

    refine_chain = REFINE_PROMPT | llm | output_parser
    refine_question = refine_chain.invoke(
        {"user_question": user_question, "user_question_analyze": user_question_analyze}
    )

    return refine_question

def clarify_user_question(
    user_question: str, user_question_analyze: str, collected_questions: List[str]
) -> str:
    output_parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=load_prompt("backend/prompts/additional_question/main_v1.prompt")
            ),
            (
                "human",
                "원래 사용자 질문:\n{user_question}\n\n초기 질문 분석:\n{user_question_analyze}\n\n이전 질문 기록:\n{collected_questions}\n\n"
                + load_prompt("backend/prompts/additional_question/human_postfix_v1.prompt"),
            ),
        ]
    )

    chain = prompt | llm | output_parser
    chat_history = "\n".join(f"{i+1}. {q}" for i, q in enumerate(collected_questions))

    leading_question = chain.invoke(
        {
            "user_question": user_question,
            "user_question_analyze": user_question_analyze,
            "collected_questions": chat_history,
        }
    )

    return leading_question


def create_query(
        user_question,
        user_question_analyze,
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
            print("user_question:", user_question)
            print("today:", today)
            # 프롬프트 로드 및 구성
            prefix = load_prompt("backend/prompts/query_creation/prefix_v1.prompt").format(
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
        list: 쿼리 결과
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
        result = cursor.fetchall()

        # 연결 닫기
        cursor.close()
        connection.close()

        print("PostgreSQL 데이터베이스 연결 종료")
        return result

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
