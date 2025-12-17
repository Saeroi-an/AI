# startpoint.py


# if __name__ == "__main__":

# @app.post("/upload") 
def handle_chat_request(session_id: str, user_query: str, image_file_path: str = None):
    
    # 1. [LOAD & PREPARE INPUT] 메모리 로드
    conversation_history = get_history_from_supabase(session_id)
    
    # 이미지 파일 경로가 있다면, 쿼리에 명시적으로 추가하여 LLM이 VL Tool을 쓰도록 유도
    if image_file_path:
        user_query_with_image = (
            f"업로드 한 다음 이미지에 대해서 '{image_file_path}', {user_query}을 답변해줘."
        )
    else:
        user_query_with_image = user_query
    
    # 최종 입력 구성 (history + query): 이전 대화 내용 + 현재 대화 내용
    full_input = (
        f"{conversation_history}\n\n" # 메모리 주입
        f"현재 사용자 요청: {user_query_with_image}"
    )

    # 2. [RUN AGENT] Agent 실행
    try:
        final_answer = AGENT_EXECUTOR.run(input=full_input)
    except Exception as e:
        final_answer = f"죄송합니다. 처리 중 오류가 발생했습니다: {e}"
        print(f"Agent Execution Error: {e}")

    # 3. [SAVE & RETURN] 메모리 저장
    save_history_to_supabase(session_id, user_query_with_image, final_answer)
    
    return {"session_id": session_id, "response": final_answer}

# # 실행 예시 (실제 앱 구동 로직):
if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000) 
    pass