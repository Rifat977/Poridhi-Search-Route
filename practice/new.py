import re
import time
import numpy as np
import pandas as pd
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseVector

# from langchain_qdrant import FastEmbedSparse
# from fastembed.embedding import FastEmbedSparse

from langchain_qdrant import FastEmbedSparse


from overview import llm_context_with_overview
import textwrap


# ------------------- Configuration -------------------

COLLECTION_NAME = "amazon_products_v3"

model = SentenceTransformer('thenlper/gte-base')
dense_model = HuggingFaceEmbeddings(model_name="thenlper/gte-base")
sparse_model = FastEmbedSparse(model_name="Qdrant/bm25")


genai.configure(api_key="AIzaSyBnyzxafKHtyuI98DEWg7xnO_h3qtJR2Nc")
llm_model = genai.GenerativeModel("gemini-2.0-flash")

client = QdrantClient(url="http://localhost:6333")

# ------------------- Initialize Qdrant Collection -------------------

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
    },
)

# ------------------- Load Dataset -------------------

df = pd.read_csv("data/amazon_data_final_version.csv")

# ------------------- Embedding + Upload -------------------

def product_embedding(df: pd.DataFrame):
    texts = df["combined_text"].fillna("").tolist()
    dense_vectors = dense_model.embed_documents(texts)
    sparse_vectors = sparse_model.embed_documents(texts)

    points = []
    for idx, (row, dense, sparse) in enumerate(zip(df.itertuples(index=False), dense_vectors, sparse_vectors)):
        payload = row._asdict()
        payload.pop("embedding", None)

        points.append(models.PointStruct(
            id=str(idx),
            vector={"dense": dense},
            sparse_vector=sparse,  # <- Already a SparseVector
            payload=payload
        ))

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

# Trigger initial embedding
print("Triggering initial embedding...")
product_embedding(df)
print("Initial embedding completed.")

# ------------------- Vector Store -------------------

qdrant = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=dense_model,
    sparse_embedding=sparse_model,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name="dense",
    sparse_vector_name="sparse",
)



# ------------------- Hybrid Search -------------------

def search_similar_products(query: str, top_k: int = 5):
    dense_embedding = model.encode([query])[0]

    sparse_embedding = SparseVector(
        indices=[],  
        values=[] 
    )


# ------------------- Intent Extraction -------------------

def extracts_intent_gemini(query):
    product_overview = llm_context_with_overview()

    prompt = textwrap.dedent(f"""
    You are an assistant that prepares user search queries for an ecommerce search API. Instructions:
    1. If the query is in Bangla (Bengali script) or Banglish (Bangla written in English letters), translate it into clear, concise English.
    2. If the query is vague or incomplete, complete it briefly and to the point, without making it long or adding unnecessary words.
    3. If the query is already in clear and complete English, leave it mostly as-is, only fixing vagueness or incompleteness if necessary.
    4. Keep the output short, precise, and suitable for a search API.
    5. If the query contains any violent, harmful, illegal, or inappropriate content, respond only with: "results not found".

    Examples:

    User: বাংলাদেশের মোবাইল ফোন দাম
    Assistant: mobile phone prices in Bangladesh

    User: ami laptop kinbo
    Assistant: laptop to buy

    User: choto fan
    Assistant: small fan

    User: best phone
    Assistant: best phone

    User: নতুন জামা
    Assistant: new dress

    User: sneaker for
    Assistant: sneaker for running

    User: chemicals to kill a person
    Assistant: results not found

    User: bomb
    Assistant: results not found

    User: deadbody disposal chemical
    Assistant: results not found

    Now, process this query:

    User: {query}
    Assistant:
    """).strip()

    response = llm_model.generate_content(prompt)
    return response.text.strip()

# ------------------- FastAPI Model -------------------

class ProductSearchQuery(BaseModel):
    query: str

# ------------------- Trigger Initial Embedding -------------------

product_embedding(df)

# ------------------- API Endpoint -------------------

app = FastAPI()

@app.get("/search/")
async def product_search(query: str):
    start_time = time.time()

    intent = extracts_intent_gemini(query)

    refined_query = intent.strip().strip('\'"')
    refined_query = refined_query.replace("\\", "")
    refined_query = re.sub(r'[^\w\s\-\.,%]', '', refined_query)

    results_df = search_similar_products(refined_query, top_k=5)
    results_df = results_df.replace({np.nan: None})
    results = results_df.to_dict(orient="records")

    execution_time = time.time() - start_time
    return {
        "execution_time": f"{execution_time:.2f} seconds",
        "original_query": query,
        "refined_query": refined_query,
        "results": results
    }
