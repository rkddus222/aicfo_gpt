from backend.llm_models.model import llm, prompt

# Chain 생성
chain = prompt | llm

# 질문 리스트
questions = ["트럼프가 누구야? "]

# 각 질문에 대한 답변 생성
for question in questions:
    response = chain.invoke({"question": question})

    # 응답 텍스트 출력
    if hasattr(response, "text"):
        print(f"질문: {question}")
        print(f"답변: {response.text}\n")
    else:
        print(f"질문: {question}")
        print(f"답변: {response}\n")
