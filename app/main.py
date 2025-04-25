# main.py

import pandas as pd
import numpy as np
import time
import re

from fastapi import FastAPI, Query
from langchain.schema import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams
import google.generativeai as genai


from typing import List, Dict, Literal
from pydantic import BaseModel

# Load CSV and prepare documents
def load_and_prepare_documents(csv_path: str):
    df = pd.read_csv(csv_path)
    docs = [
        Document(
            page_content=row["combined_text"],
            metadata={
                "product_id": row["asin"],
                "title": row["title"],
                "brand": row["brand"],
                "description": row["description"],
            }
        )
        for _, row in df.iterrows()
    ]
    return docs

def initialize_vector_store(docs, collection_name="amazon_products_final", local_path="/home/rifat/qdrant_data"):
    print("⚙️ Initializing Qdrant vector store...")
    dense = HuggingFaceEmbeddings(model_name="thenlper/gte-base")
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")
    client = QdrantClient(url="http://localhost:6333")

    try:
        collection_info = client.get_collection(collection_name)
        print(f"✅ Collection '{collection_name}' found. Skipping creation.")
    except Exception as e:
        print(f"❗ Collection '{collection_name}' not found. Creating new collection...")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=768, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            },
        )
        print("➕ Collection created successfully.")

        ids = list(range(1, len(docs) + 1))
        
        print("📥 Adding documents to Qdrant with assigned integer IDs...")
        qdrant_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=dense,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        qdrant_store.add_documents(documents=docs, ids=ids)
        print("✅ Vector store initialized with explicit IDs.")

    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse"
    )
    print("✅ Vector store initialized and ready.")
    return qdrant_store



# Configure Gemini
def setup_gemini(api_key: str):
    print("🔐 Configuring Gemini API...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    print("✅ Gemini model ready.")
    return model

# Query optimization
def extracts_intent_gemini(raw_query: str) -> str:
    prompt_template = """
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

    User: {}
    Assistant:
    """.strip()
    full_prompt = prompt_template.format(raw_query)
    response = gemini_model.generate_content(full_prompt)
    return response.text.strip()

# Run semantic search
def search_similar_products(query: str, top_k: int = 5) -> pd.DataFrame:
    print(f"🔎 Running hybrid vector search for: {query}")
    results = qdrant_store.similarity_search_with_score(query=query, k=top_k)
    output = []
    for doc, score in results:
        item = doc.metadata.copy()
        item["similarity_score"] = score
        print(f"➡️  [SIM={score:.3f}] {item['title']}")
        output.append(item)
    return pd.DataFrame(output)

# Load data and initialize once
docs = load_and_prepare_documents("data/amazon_data_final_version.csv")
qdrant_store = initialize_vector_store(docs)
gemini_model = setup_gemini("AIzaSyBnyzxafKHtyuI98DEWg7xnO_h3qtJR2Nc")

# Create FastAPI app
app = FastAPI()

@app.get("/search/")
async def product_search(query: str = Query(..., description="User's product search query")):
    start_time = time.time()
    print(f"\n📥 New search request: {query}")

    intent = extracts_intent_gemini(query)

    refined_query = intent.strip().strip('\'"').replace("\\", "")
    refined_query = re.sub(r'[^\w\s\-\.,%]', '', refined_query)
    print(f"🧼 Cleaned query: {refined_query}")

    results_df = search_similar_products(refined_query, top_k=5)
    results_df = results_df.replace({np.nan: None})
    results = results_df.to_dict(orient="records")

    execution_time = time.time() - start_time
    print(f"✅ Done in {execution_time:.2f}s with {len(results)} results.\n")

    return {
        "execution_time": f"{execution_time:.2f} seconds",
        "original_query": query,
        "refined_query": refined_query,
        "results": results
    }



class ProductInput(BaseModel):
    product_id: int
    title: str
    brand: str
    description: str
    combined_text: str

class TriggerPayload(BaseModel):
    action: Literal["update"]
    products: List[ProductInput]

@app.post("/trigger")
async def trigger_vector_update(payload: TriggerPayload):
    print("🚨 Trigger received: Vector update")

    if payload.action != "update":
        return {"status": "error", "message": "Unsupported action"}

    updated_docs = []
    ids_to_update = []

    for product in payload.products:
        print(f"🔁 Processing product ID: {product.product_id}")

        # Attempt to delete the existing vector based on product_id
        try:
            # Fix: Delete using PointIdsList with integer point IDs
            qdrant_store.client.delete(
                collection_name=qdrant_store.collection_name,
                points_selector=models.PointIdsList(points=[int(product.product_id)])  # Ensure it's an integer
            )
            print(f"🗑️ Deleted existing vector ID: {product.product_id}")
        except Exception as e:
            print(f"⚠️ Could not delete ID {product.product_id} (maybe new): {e}")

        # Create the document to be added or updated
        doc = Document(
            page_content=product.combined_text,
            metadata={
                "product_id": product.product_id,
                "title": product.title,
                "brand": product.brand,
                "description": product.description
            }
        )
        updated_docs.append(doc)
        ids_to_update.append(int(product.product_id))  # Ensure it's an integer ID

    if updated_docs:
        print("📥 Adding new/updated documents to vector store...")
        qdrant_store.add_documents(documents=updated_docs, ids=ids_to_update)
        print("✅ Update complete.")

    return {"status": "success", "message": f"{len(updated_docs)} product(s) updated"}
