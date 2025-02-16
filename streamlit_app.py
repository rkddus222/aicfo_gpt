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

    st.header("AICFO GPT")

    # 사용자 입력 받기
    task = st.text_input("질문을 입력해주세요.")

    if st.button("입력"):
        if task:
            try:
                # GraphState 입력값 구성
                graph_input = {
                    "user_question": task,
                    "selected_question": "",
                    "user_question_eval": "",
                    "user_question_analyze": "",
                    "final_answer": "",
                    "sql_query": "",
                    "query_result": [],
                }

                # Graph invoke 호출
                with st.spinner("잠시만 기다려주세요..."):
                    data = graph.invoke(graph_input)

                # Graph 결과 확인
                final_answer = data.get("final_answer", "No final answer returned.")
                sql_query = data.get("sql_query", "No sql_query returned.")
                query_result = data.get("query_result", {"columns": [], "rows": []})

                st.subheader("결과")
                st.write(sql_query)
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
