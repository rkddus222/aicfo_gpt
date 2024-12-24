import streamlit as st
from backend.langgraph_.graph import make_graph
import pandas as pd


def main():
    try:
        # Graph 초기화
        graph = make_graph()
    except Exception as e:
        st.error(f"Graph 초기화 오류: {str(e)}")
        return

    st.header("DAQUV LLM")

    # 사용자 입력 받기
    task = st.text_input("질문을 입력해주세요.")

    if st.button("입력"):
        if task:
            try:
                # GraphState 입력값 구성
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
                with st.spinner("잠시만 기다려주세요..."):
                    data = graph.invoke(graph_input)

                # Graph 결과 확인
                final_answer = data.get("final_answer", "No final answer returned.")
                sql_query = data.get("sql_query", "No sql_query returned.")
                query_result = data.get("query_result", {"columns": [], "rows": []})

                st.subheader("결과")
                st.write(final_answer)

                # Query 결과를 DataFrame으로 표시
                if query_result and query_result.get("rows"):
                    st.subheader("상세 결과")
                    st.write(sql_query)
                    columns = query_result.get("columns", [])
                    rows = query_result.get("rows", [])

                    # DataFrame 변환 및 표시
                    df = pd.DataFrame(rows, columns=columns)
                    df.index = df.index + 1
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a task.")


if __name__ == "__main__":
    main()
