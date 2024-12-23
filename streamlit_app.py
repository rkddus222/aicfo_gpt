import streamlit as st
from backend.langgraph_.graph import make_graph


def main():
    try:
        graph = make_graph()
    except Exception as e:
        st.error(f"Graph 초기화 오류: {str(e)}")
        return

    st.header("DAQUV LLM")

    task = st.text_input("Enter your task:")

    if st.button("Process"):
        if task:
            try:
                # GraphState 입력값 확인
                graph_input = {
                    "llm_api": "ChatGPT-4o",
                    "user_question": task,
                    "user_question_eval": "",
                    "user_question_analyze": "",
                    "collected_questions": [],
                    "ask_user": 0,
                    "final_answer": "",
                    "context_cnt": 0,
                    "table_contexts": [],
                    "table_contexts_ids": [],
                    "need_clarification": False,
                    "sample_info": 0,
                    "sql_query": "",
                    "flow_status": "KEEP",
                    "max_query_fix": 3,
                    "query_fix_cnt": 0,
                    "query_result": [],
                    "error_msg": ""
                }

                # Graph invoke 호출
                with st.spinner("Processing..."):
                    data = graph.invoke(graph_input)

                # 반환값 확인
                st.subheader("Graph Output")
                st.json(data)

                # 결과 확인
                final_answer = data.get("final_answer", "No final answer returned.")
                sql_query = data.get("sql_query", "No sql_query returned.")
                st.subheader("결과")
                st.write(sql_query)
                st.write(final_answer)

            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a task.")


if __name__ == "__main__":
    main()
