from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import trafilatura
import openai
import faiss
import numpy as np
import sqlite3
import pickle
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

# --- Basic Auth setup ---
security = HTTPBasic()
USER = os.getenv("API_USER", "admin")
PASS = os.getenv("API_PASS", "6434e108a8efccf2e8629862b70af80f")

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != USER or credentials.password != PASS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- OpenAI setup ---
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Missing OpenAI API key. Make sure OPENAI_API_KEY is set.")

# --- Persistent storage setup ---
storage_dir = os.getenv("STORAGE_DIR", "/opt/render/project/src/storage")
os.makedirs(storage_dir, exist_ok=True)
DB_PATH = os.path.join(storage_dir, "chatbot_data.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
# Domains table
c.execute(
    '''CREATE TABLE IF NOT EXISTS domains (
        domain TEXT PRIMARY KEY,
        index_blob BLOB,
        chunks_blob BLOB,
        urls_blob BLOB
    )'''
)
# Queries log table
c.execute(
    '''CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT,
        question TEXT,
        answer TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )'''
)
conn.commit()

# --- FastAPI setup ---
app = FastAPI(docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class DomainRequest(BaseModel):
    domain: str

class QueryRequest(BaseModel):
    domain: str
    question: str

# --- In-memory stores ---
index_store = {}
chunks_store = {}  # { domain: {'chunks': [...], 'urls': [...] } }
urls_store = {}

# --- Helpers ---
def normalize(domain: str) -> str:
    return domain.replace("https://", "").replace("http://", "").replace("www.", "").strip("/")

def fetch_internal_links(base_url: str, max_links: int = 20) -> list[str]:
    try:
        resp = requests.get(base_url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        links = {base_url}
        netloc = urlparse(base_url).netloc
        for tag in soup.find_all("a", href=True):
            href = urljoin(base_url, tag["href"])
            p = urlparse(href)
            if p.netloc == netloc and p.scheme.startswith("http"):
                links.add(href)
            if len(links) >= max_links:
                break
        return list(links)
    except:
        return [base_url]

# --- Create Chatbot ---
@app.post("/create-chatbot")
async def create_chatbot(req: DomainRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    base_url = f"https://{dom}"
    urls = fetch_internal_links(base_url)
    idx = faiss.IndexFlatL2(1536)
    chunks = []
    chunk_urls = []
    for url in urls:
        raw = trafilatura.fetch_url(url)
        text = trafilatura.extract(raw) if raw else None
        if not text:
            continue
        for i in range(0, len(text), 1000):
            chunk = text[i:i+1000]
            emb = openai.embeddings.create(input=chunk, model="text-embedding-3-small").data[0].embedding
            idx.add(np.array([emb]).astype('float32'))
            chunks.append(chunk)
            chunk_urls.append(url)
    index_store[dom] = idx
    chunks_store[dom] = {'chunks': chunks, 'urls': chunk_urls}
    # Persist
    blob_idx = pickle.dumps(idx)
    blob_chunks = pickle.dumps(chunks)
    blob_urls = pickle.dumps(chunk_urls)
    c.execute(
        "INSERT OR REPLACE INTO domains (domain, index_blob, chunks_blob, urls_blob) VALUES (?, ?, ?, ?)",
        (dom, blob_idx, blob_chunks, blob_urls)
    )
    conn.commit()
    return {"chatbot_url": f"/ {dom.replace('.', '-')}" , "indexed": True, "fetched_urls": urls}

# --- Ask Bot ---
@app.post("/ask")
async def ask_bot(req: QueryRequest, user: str = Depends(get_current_user)):
    dom = normalize(req.domain)
    if dom not in index_store:
        c.execute("SELECT index_blob, chunks_blob, urls_blob FROM domains WHERE domain = ?", (dom,))
        row = c.fetchone()
        if row:
            index_store[dom] = pickle.loads(row[0])
            chunks_store[dom] = {'chunks': pickle.loads(row[1]), 'urls': pickle.loads(row[2])}
    idx = index_store.get(dom)
    store = chunks_store.get(dom, {})
    chunks = store.get('chunks', [])
    chunk_urls = store.get('urls', [])
    if not idx or not chunks:
        return {"answer": "No content indexed yet for this domain. Please create a chatbot first."}
    user_emb = openai.embeddings.create(input=req.question, model="text-embedding-3-small").data[0].embedding
    D, I = idx.search(np.array([user_emb]).astype('float32'), k=3)
    selected_chunks = []
    selected_urls = []
    for i in I[0]:
        if i < len(chunks):
            selected_chunks.append(chunks[i])
            selected_urls.append(chunk_urls[i])
    if not selected_chunks:
        return {"answer": "Sorry, I couldn't find relevant content to answer your question."}
    context = "\n---\n".join(selected_chunks)
    prompt = f"Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {req.question}"
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Provide clickable links to source pages in your response."},
            {"role": "user", "content": prompt}
        ]
    )
    answer_text = completion.choices[0].message.content.strip()
    # Log query
    c.execute(
        "INSERT INTO queries (domain, question, answer) VALUES (?, ?, ?)",
        (dom, req.question, answer_text)
    )
    conn.commit()
    # unique URLs
    references = list(dict.fromkeys(selected_urls))
    return {"answer": answer_text, "references": references}

# --- Proxy endpoints for client ---
@app.post("/client/create-chatbot")
async def client_create(req: DomainRequest):
    return await create_chatbot(req)

@app.post("/client/ask")
async def client_ask(req: QueryRequest):
    return await ask_bot(req)

# --- Other endpoints omitted for brevity ---
