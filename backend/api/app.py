from typing import List, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from models import URLRequest, QueryRequest, QueryResponse, YT_URL
from fetcher import load_urls, process_uploaded_file
from yt_audio_fetcher import process_audio_upload, process_youtube_upload
from embed_data import embed_documents, clear_vectordb
from rag_qa import ArticleQAEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods=["*"],
    allow_headers=["*"]
)

vector_path = "faiss_store.pkl"
qa = ArticleQAEngine(vector_path=vector_path)


@app.post("/process-urls/")
def process_urls(request: URLRequest):
    try:
        # url_vector_path = "faiss_store.pkl"
        docs = load_urls(request.urls)
        embed_documents(docs=docs, VECTOR_DIR=vector_path)        
        qa.set_retriever_from_local(vector_path=vector_path)
        return {'status':'success', 'message':'URLs Processed & Embedded'}
    except Exception as e:
        return {'status': "error", 'message':str(e)}
   
@app.post("/process-yt/")
def process_yt(request: YT_URL):
    print(request.model_dump())
    try:
        docs = process_youtube_upload(request.yt_url)
        embed_documents(docs=docs, VECTOR_DIR=vector_path)        
        qa.set_retriever_from_local(vector_path=vector_path)
        return {'status':'success', 'message':'YT URL Processed & Embedded'}
    except Exception as e:
        return {'status': "error", 'message':str(e)}

@app.post("/process-audio/")
def process_audio(file: UploadFile = File()):
    try:
        # audio_vector_path = "faiss_store.pkl"
        docs = process_audio_upload(file)
        embed_documents(docs=docs, VECTOR_DIR=vector_path)        
        qa.set_retriever_from_local(vector_path=vector_path)
        return {'status':'success', 'message':'Audio Processed & Embedded'}
    except Exception as e:
        return {'status': "error", 'message':str(e)}

@app.post("/process-file/")
async def process_file(file: UploadFile = File()):
    try:
        docs = await process_uploaded_file(file)
        embed_documents(docs=docs,VECTOR_DIR=vector_path)
        qa.set_retriever_from_local(vector_path=vector_path)        
        return {"status": "success", "message": "Uploaded file processed and embedded."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/ask/", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        answer = qa.answer_question(query=request.question, chat_history=request.chat_history)
        return QueryResponse(answer=answer)
    except ValueError as e:
        return QueryResponse(answer=str(e))
        
@app.post("/reset/")
def reset_engine():
    global qa
    clear_vectordb()  # Fully clears stored vectors and article data
    qa = ArticleQAEngine(vector_path=vector_path)  # Reinitialize to reflect cleared store
    return {"status": "success", "message": "Reset successful. QA engine reloaded. Please reprocess articles."}
