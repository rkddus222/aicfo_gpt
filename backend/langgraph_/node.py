import os
from typing import TypedDict, Any, Dict
from .task import (
    evaluate_user_question,
    simple_conversation,
    create_query,
    analyze_user_question,
    business_conversation,
    execute_query, extract_table_name_from_text,
)
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# GrpahState 정의
class GraphState(TypedDict):
    # 그래프 내에서 사용될 모든 key값을 정의해야 오류가 나지 않는다.
    user_question: str  # 사용자의 질문
    selected_table : str
    user_question_eval: str  # 사용자의 질문이 SQL 관련 질문인지 여부
    user_question_analyze: str  # 사용자 질문 분석
    final_answer: str
    sql_query: str
    query_result: Dict[str, Any]


########################### 정의된 노드 ###########################
def question_evaluation(state: GraphState) -> GraphState:
    print("DEBUG - question_evaluation: 시작", state)
    user_question = state.get("user_question", "")
    user_question_eval = evaluate_user_question(user_question)
    print("DEBUG - question_evaluation: 평가 결과", user_question_eval)

    state.update({"user_question_eval": user_question_eval})
    return state


def non_sql_conversation(state: GraphState) -> GraphState:
    user_question = state["user_question"]
    final_answer = simple_conversation(user_question)

    state.update({"final_answer": final_answer})
    print("DEBUG - non_sql_conversation: 시작", state)
    return state


def question_analyze(state: GraphState) -> GraphState:
    print("DEBUG - question_analyze: 시작", state)
    user_question = state["user_question"]
    analyze_question = analyze_user_question(user_question)

    table_name = extract_table_name_from_text(analyze_question)
    state.update({"user_question_analyze": analyze_question, "selected_table": table_name})
    return state

def query_creation(state: GraphState) -> GraphState:
    """
    사용자 질문을 기반으로 SQL 쿼리를 생성하고 상태를 업데이트하는 노드.

    Args:
        state (GraphState): 그래프 상태

    Returns:
        GraphState: 업데이트된 그래프 상태
    """
    print("DEBUG - query_creation: 시작", state)
    user_question = state["user_question"]
    user_question_analyze = state["user_question_analyze"]
    selected_table = state["selected_table"]
    today = datetime.now().strftime("%Y-%m-%d")

    # SQL 쿼리 생성
    sql_query = create_query(user_question, user_question_analyze, selected_table, today)
    print("sql_query: ", sql_query)
    # 상태 업데이트
    state.update({
        "sql_query": sql_query,
    })
    print("state: ", state)
    return state

def get_query_result(state: GraphState) -> GraphState:
    # 환경 변수에서 DB 정보 가져오기
    host = os.getenv("DB_HOST")
    database = os.getenv("DB_DATABASE")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port = os.getenv("DB_PORT")

    # SQL 쿼리 가져오기
    query = state.get("sql_query")
    if not query:
        raise ValueError("SQL 쿼리가 state에 포함되어 있지 않습니다.")

    # DB 쿼리 실행
    result = execute_query(host, database, user, password, port, query)

    print("쿼리 결과", result)

    # 결과가 None인 경우 빈 리스트로 초기화
    if result is None or not result["rows"]:
        result = {"columns": [], "rows": []}

    # 결과가 너무 많을 경우 상위 100개로 제한
    if len(result["rows"]) > 100:
        result["rows"] = result["rows"][:100]

    # 상태 업데이트
    state.update({
        "query_result": result
    })
    return state



def sql_conversation(state: GraphState) -> GraphState:
    user_question = state["user_question"]
    sql_query = state["sql_query"]
    query_result = state.get("query_result", [])
    final_answer = business_conversation(
        user_question, sql_query=sql_query, query_result=query_result
    )

    state.update({"final_answer": final_answer})
    return state



################### ROUTERS ###################
def user_question_checker(state: GraphState) -> str:
    """그래프 상태에서 사용자의 질문 분류 결과를 가져오는 노드입니다.

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        str: 사용자의 질문 분류 결과 ("1" or "0")
    """
    return state["user_question_eval"]