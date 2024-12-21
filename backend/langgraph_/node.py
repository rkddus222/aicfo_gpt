import os
from typing import TypedDict, List, Any
from .task import (
    evaluate_user_question,
    simple_conversation,
    create_query,
    analyze_user_question,
    business_conversation,
    clarify_user_question,
    check_leading_question,
    refine_user_question,
    execute_query,
)
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# FAISS 객체는 serializable 하지 않아 Graph State에 넣어 놓을 수 없다.
from .faiss_init import get_vector_stores


# GrpahState 정의
class GraphState(TypedDict):
    # Warning!
    # 그래프 내에서 사용될 모든 key값을 정의해야 오류가 나지 않는다.
    llm_api: str  # Local, ChatGPT-4o
    user_question: str  # 사용자의 질문
    user_question_eval: str  # 사용자의 질문이 SQL 관련 질문인지 여부
    user_question_analyze: str  # 사용자 질문 분석
    collected_questions: List[str]  # 사용자의 질문에 대한 추가 질문-대답 기록
    ask_user: int  # leading question 질문 여부 [0, 1]
    final_answer: str
    # TODO
    # context_cnt가 동적으로 조절 되도록 알고리즘을 짜야 한다.
    context_cnt: int  # 사용자의 질문에 대답하기 위해서 정보를 가져올 context 갯수
    table_contexts: List[str]
    table_contexts_ids: List[int]
    need_clarification: bool  # 사용자 추가 질문(설명)이 필요한지 여부
    sample_info: int
    sql_query: str
    flow_status: str  # KEEP, REGENERATE, RE-RETRIEVE, RESELECT, RE-REQUEST
    max_query_fix: int
    query_fix_cnt: int
    query_result: List[Any]
    error_msg: str


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
    user_question = state["user_question"]
    analyze_question = analyze_user_question(user_question)

    state.update({"user_question_analyze": analyze_question})
    print("DEBUG - question_analyze: 시작", state)
    return state

def question_clarify(state: GraphState) -> GraphState:
    """사용자 질문이 모호할 경우 추가 질문을 통해 질문 분석을 진행하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문을 분석한 대답이 추가된 그래프 상태
    """
    user_question_analyze = state["user_question_analyze"]
    user_question = state["user_question"]
    collected_questions = state.get("collected_questions", [])

    # 사용자 질문 명확화 함수 호출
    leading_question = clarify_user_question(
        user_question, user_question_analyze, collected_questions
    )
    ask_user = check_leading_question(leading_question)

    # 기존 상태를 업데이트
    state.update({
        "collected_questions": collected_questions + [leading_question],
        "ask_user": ask_user
    })
    print("DEBUG - question_clarify: 시작", state)
    return state


def human_feedback(state: GraphState) -> GraphState:
    # 상태에 변화가 없으면 필수 키 중 하나를 더미 값으로 업데이트
    state.update({"flow_status": state.get("flow_status", "KEEP")})
    return state

def question_refine(state: GraphState) -> GraphState:
    """질문 구체화를 진행하는 노드

    Args:
        state (GraphState): LangGraph에서 쓰이는 그래프 상태

    Returns:
        GraphState: 사용자의 질문에 대한 대답이 추가된 그래프 상태
    """
    collected_questions = state["collected_questions"]
    user_question_analyze = collected_questions[-1]
    user_question = state["user_question"]

    # 사용자 질문을 구체화
    refine_question = refine_user_question(user_question, user_question_analyze)

    # 기존 상태를 업데이트
    state.update({
        "user_question": refine_question
    })
    print("DEBUG - question_refine: 시작", state)
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
    today = datetime.now().strftime("%Y-%m-%d")
    flow_status = state.get("flow_status", "KEEP")

    # SQL 쿼리 생성
    sql_query = create_query(user_question, user_question_analyze, today)
    print("sql_query: ", sql_query)
    # 상태 업데이트
    state.update({
        "sql_query": sql_query,
    })
    print("state: ", state)
    return state

def get_query_result(state: GraphState) -> GraphState:
    host = os.getenv("DB_HOST")
    database = os.getenv("DB_DATABASE")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port = os.getenv("DB_PORT")
    query = state["sql_query"]
    result = execute_query(host, database, user, password, port, query)

    if not query:
        raise ValueError("SQL 쿼리가 state에 포함되어 있지 않습니다.")

    if len(result) > 100:
        result = result[:100]

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


def user_question_analyze_checker(state: GraphState) -> bool:
    user_question_analyze = state["user_question_analyze"]
    analyze_question = analyze_user_question(user_question_analyze)

    keywords = ["[에러]"]
    return any(keyword in analyze_question for keyword in keywords)



def leading_question_checker(state: GraphState) -> str:
    ask_user = state["ask_user"]
    if ask_user == 0:
        return "ESCAPE"
    else:
        return "KEEP"

