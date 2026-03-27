# Traditional Vector RAG Application

This repository contains a full-stack, 4-step Retrieval-Augmented Generation (RAG) application utilizing traditional vector embeddings.

## Architecture

The application implements a strict 4-step pipeline to visualize the traditional RAG workflow:
1. **Document Upload & Chunking:** Processes PDF files using LangChain's `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
2. **Create Embeddings:** Generates vector embeddings via OpenAI models and stores them in a `DocArrayInMemorySearch` database.
3. **Similarity Search:** Queries the vector database to retrieve the top $K$ semantic matches from the chunks.
4. **Answer Generation:** Passes the retrieved context chunks and the initial query to a `ChatOpenAI` LLM to formulate a reasoned answer.

### Tech Stack
- **Frontend:** React + Vite (Glassmorphism layout)
- **Backend:** FastAPI, LangChain, DocArray, OpenAI

---

## Setup Instructions

### 1. Backend Configuration
1. Navigate to the backend directory:
   ```sh
   cd backend
   ```
2. Create your virtual environment and install dependencies:
   ```sh
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the `backend/` directory:
   ```
   OPENAI_API_KEY=sk-your-openai-api-key
   ```
4. Run the API Server:
   ```sh
   python main.py
   ```
   *The server runs on `http://localhost:8001`.*

### 2. Frontend Configuration
1. Navigate to the frontend directory:
   ```sh
   cd frontend
   ```
2. Install npm packages:
   ```sh
   npm install
   ```
3. Run the development server:
   ```sh
   npm run dev -- --port 5174
   ```
   *The React UI runs on `http://localhost:5174`.*

---

## Usage

1. Open `http://localhost:5174` in your browser.
2. Upload any PDF document to securely split and preview the generated raw text chunks.
3. Click to process embeddings and hold them in the temporary DocArray database.
4. Test out queries and explicitly see the vectors returned during similarity search!
5. Generate LLM answers grounded in those vectors.
