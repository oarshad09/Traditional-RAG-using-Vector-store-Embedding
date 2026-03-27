import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# LangChain imports provided by the user
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI()

# Allow CORS for local frontend tests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to simulate database and state
GLOBAL_STATE = {
    "docs": [],
    "db": None,
    "qa_chain": None,
    "chat_history": []
}

class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Save the file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        # STEP 1: Load and split the document
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        
        # Store in state
        GLOBAL_STATE["docs"] = docs
        
        # Cleanup temporary file
        os.unlink(tmp_path)
        
        return {
            "status": "success", 
            "message": f"Document loaded and split into {len(docs)} chunks.",
            "sample_chunks": [doc.page_content[:200] + "..." for doc in docs[:3]]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed")
async def create_embeddings():
    docs = GLOBAL_STATE.get("docs")
    if not docs:
        raise HTTPException(status_code=400, detail="No documents uploaded or parsed yet.")
        
    try:
        # STEP 2: Create embeddings and store in DocArrayInMemorySearch
        embeddings = OpenAIEmbeddings()
        db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        
        # Define retriever explicitly (as in the provided code)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Create the LLM instead of legacy chain
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        GLOBAL_STATE["db"] = db
        GLOBAL_STATE["llm"] = llm
        GLOBAL_STATE["retriever"] = retriever
        GLOBAL_STATE["chat_history"] = []
        
        return {"status": "success", "message": f"Successfully created embeddings for {len(docs)} chunks and stored in vector database."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def similarity_search(request: QueryRequest):
    db: DocArrayInMemorySearch = GLOBAL_STATE.get("db")
    if not db:
        raise HTTPException(status_code=400, detail="Vector search database not initialized. Please embed a document first.")
        
    try:
        # STEP 3: Perform a similarity search
        results = db.similarity_search(request.query, k=4)
        
        nodes = []
        for i, doc in enumerate(results):
            nodes.append({
                "node_id": f"chunk-{i+1}",
                "page": doc.metadata.get("page", 0),
                "text": doc.page_content
            })
            
        return {
            "status": "success",
            "thinking": f"Performed embedding distance metric comparison on '{request.query}' against the DocArray index.",
            "nodes": nodes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_answer(request: QueryRequest):
    llm = GLOBAL_STATE.get("llm")
    retriever = GLOBAL_STATE.get("retriever")
    if not llm or not retriever:
        raise HTTPException(status_code=400, detail="LLM or Retriever not initialized. Please embed a document first.")
        
    try:
        # STEP 4: Generate the final answer using manual retrieval and inference
        docs = retriever.invoke(request.query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"Answer the following question based ONLY on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{request.query}\n\nAnswer:"
        
        # Invoke LLM
        result = llm.invoke(prompt)
        
        # Update chat history manually 
        GLOBAL_STATE["chat_history"].append((request.query, result.content))
        
        return {
            "status": "success",
            "answer": result.content,
            "sources": [doc.page_content for doc in docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
