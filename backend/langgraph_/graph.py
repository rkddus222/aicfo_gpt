from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .node import (
    GraphState,
    question_evaluation,
    non_sql_conversation,
    user_question_checker,
    query_creation,
    question_analyze,
    human_feedback,
    question_clarify,
    user_question_analyze_checker,
    leading_question_checker,
    sql_conversation,
    question_refine,
    get_query_result,
)


def make_graph() -> CompiledStateGraph:
    workflow = StateGraph(GraphState)

    # 노드 추가
    workflow.add_node("question_evaluation", question_evaluation)
    workflow.add_node("general_conversation", non_sql_conversation)
    workflow.add_node("question_analysis", question_analyze)
    workflow.add_node("additional_questions", question_clarify)  # 추가 질문
    workflow.add_node("question_refinement", question_refine)  # 질문 구체화
    workflow.add_node("sql_query_generation", query_creation)
    workflow.add_node("sql_query_result", get_query_result)
    workflow.add_node("response", sql_conversation)
    workflow.add_node("human_feedback", human_feedback)

    # 조건부 엣지 설정
    workflow.add_conditional_edges(
        "question_evaluation",
        user_question_checker,
        {
            "0": "general_conversation",  # 일반 대화일 경우 non_sql_conversation
            "1": "question_analysis",  # SQL 관련 질문 분석
        },
    )


    workflow.add_edge("question_analysis", "additional_questions")
    workflow.add_edge("additional_questions", "sql_query_generation")
    workflow.add_edge("sql_query_generation", "sql_query_result")
    workflow.add_edge("sql_query_result", "response")


    workflow.add_edge("response", END)

    # Entry point 설정
    workflow.set_entry_point("question_evaluation")

    # 그래프 컴파일
    app = workflow.compile()

    return app
