from fastapi import FastAPI, File, UploadFile, HTTPException
from model import QueryInput, QueryResponse
from database import create_application_logs,insert_application_logs,get_chat_history
from rag_chain import get_ragchain
import os
import uuid
import logging
import shutil


logging.basicConfig(filename='app.log', level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

@app.post("/", response_model=QueryResponse)
def chat(query_input: QueryInput):
    create_application_logs()
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")

    chat_history = get_chat_history(session_id)
    rag_chain = get_ragchain(query_input.model.value)
    answer = rag_chain.invoke({"input": query_input.question, "chat_history": chat_history})['answer']
    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)



# if __name__ == "__main__":
#     create_application_logs()
#     session_id = str(uuid.uuid4())
#     question = "What is Pubblika?"
#     chat_history = get_chat_history(session_id)
#     answer = rag_chain.invoke({"input": question, "chat_history": chat_history})['answer']
#     insert_application_logs(session_id, question, answer, "gemini-1.5-flash")
#     print(f"Human: {question}")
#     print(f"AI: {answer}\n")

#     # Example of a follow-up question
#     question2 = "What are Pubblikas services?"
#     chat_history = get_chat_history(session_id)
#     answer2 = rag_chain.invoke({"input": question2, "chat_history": chat_history})['answer']
#     insert_application_logs(session_id, question2, answer2, "gemini-1.5-flash")
#     print(f"Human: {question2}")
#     print(f"AI: {answer2}")

#     print(chat_history)