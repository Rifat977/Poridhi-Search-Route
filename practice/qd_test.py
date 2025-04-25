import faiss, re, time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import google.generativeai as genai

from qdrant_client import QdrantClient
from qdrant_client.http import models

from qdrant_client.http.models import SearchParams

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from contextlib import asynccontextmanager

model = SentenceTransformer('all-MiniLM-L6-v2')

api_key = "AIzaSyBnyzxafKHtyuI98DEWg7xnO_h3qtJR2Nc"
genai.configure(api_key=api_key)
llm_model = genai.GenerativeModel("gemini-2.0-flash")

df = pd.read_csv("data/final_data.csv")

COLLECTION_NAME = "amazon_products"

client = AsyncQdrantClient(
    url="http://localhost:6333",
    prefer_grpc=True,
    timeout=5.0,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    await client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        ),
    )
    print("✅ Qdrant collection ready")
    yield

app = FastAPI(lifespan=lifespan)


# ---------- Step 1: Create and Save Embeddings ----------
def product_embedding(df):
    # 1) Compute embeddings
    embeddings = model.encode(
        df.apply(
            lambda row: f"""
                {row['title']} by {row['brand']}. 
                Category: {row['primary_category']}. 
                Description: {row['description']}. 
                Price: ${row['final_price']} ({row['price_bucket']} price range). 
                Customer Rating: {row['rating']} stars. 
                Weight: {round(row['item_weight'], 2)}grams. 
                Dimensions: {round(row['length'], 2)} cm x {round(row['width'], 2)} cm x {round(row['height'], 2)} cm. 
                Department: {row['department']}. 
            """, axis=1
        ),
    )

    # 2) Prepare Qdrant points
    points = []
    for idx, row in df.iterrows():
        payload = row.to_dict()
        # remove fields we don't want in payload
        payload.pop("embedding", None)
        points.append(models.PointStruct(
            id=idx,
            vector=embeddings[idx],
            payload=payload
        ))

    # 3) Upload to Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )


# ---------- Step 2: Search Similar Products ----------
async def search_similar_products(query: str, top_k: int = 5):
    # 1) Embed the incoming query
    q_vec = model.encode([query])[0]

    # 2) Ask Qdrant for the top_k most similar vectors
    # search_result = client.search(
    #     collection_name=COLLECTION_NAME,
    #     query_vector=q_vec,
    #     limit=top_k,
    #     with_payload=True,
    #     with_vectors=False,    # we don’t need the raw vectors back
    # )

    search_result = await client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vec,
        limit=5,
        with_payload=True,
        search_params=SearchParams(hnsw_ef=128),  # higher ef → better recall
    )

    # 3) Build a DataFrame of hits
    records = []
    for hit in search_result:
        rec = hit.payload
        rec["similarity_score"] = float(hit.score)  # e.g. cosine similarity
        records.append(rec)

    return pd.DataFrame(records)


from overview import llm_context_with_overview
import textwrap

def extracts_intent_gemini(query):

    product_overview = llm_context_with_overview()


    prompt = textwrap.dedent(f"""
    You are a product query optimization engine for an e-commerce search system.

    Your job is to refine vague product queries by replacing terms like "choto", "cheap", "light", "heavy", or "expensive" with **realistic values** using statistical insights from the product catalog.

    --- Product Overview ---
    {product_overview}
    ------------------------

    ### Instructions:
    - Rewrite only if vague terms are present.
    - Use the statistical context to infer realistic ranges:
      - Price → USD
      - Weight → grams (realistic: fans = hundreds to thousands of grams)
      - Dimensions → cm
    - DO NOT invent values: base them on catalog statistics.
    - NEVER use unrealistic values (e.g., a fan weighing 10 grams).
    - If the query is already specific, return it as-is.
    - Output only the rewritten query, **no labels, no explanations**.

    ### Examples:
    Input: "cheap smartwatch"
    Output: "smartwatch under 30 USD"

    Input: "choto toothpaste"
    Output: "toothpaste under 20 grams"

    Input: "choto lightweight fan"
    Output: "lightweight fan under 1000 grams"

    Input: "wireless headphones"
    Output: "wireless headphones"

    ### Now rewrite the query below:
    "{query}"
    """)

    response = llm_model.generate_content(prompt)
    return response.text.strip()

# ---------- Main Flow ----------
class ProductSearchQuery(BaseModel):
    query: str  


# Generate embeddings and FAISS index
product_embedding(df)

@app.get("/search/")
async def product_search(query: str):
    start_time = time.time()

    intent = extracts_intent_gemini(query)

    refined_query = intent.strip().strip('\'"')         
    refined_query = refined_query.replace("\\", "")    
    refined_query = re.sub(r'[^\w\s\-\.,%]', '', refined_query) 

    results_df = await search_similar_products(refined_query, top_k=5)
    if results_df.empty:
        raise HTTPException(status_code=404, detail="No products found.")
    results_df = results_df.drop(columns=["embedding"])
    results_df = results_df.rename(columns={
        "title": "Title",
        "brand": "Brand",
        "primary_category": "Category",
        "description": "Description",
        "final_price": "Price",
        "rating": "Rating",
        "item_weight": "Weight (grams)",
        "length": "Length (cm)",
        "width": "Width (cm)",
        "height": "Height (cm)",
        "department": "Department",
        "similarity_score": "Similarity Score"
    })
    results_df = results_df[[
        "Title", "Brand", "Category", "Description", "Price",
        "Rating", "Weight (grams)", "Length (cm)", "Width (cm)",
        "Height (cm)", "Department", "Similarity Score"
    ]]
    results_df = results_df.sort_values(by="Similarity Score", ascending=False)


    print(results_df)



    results_df = results_df.replace({np.nan: None})
    results = results_df.to_dict(orient="records")

    execution_time = time.time() - start_time
    return {
        "execution_time": f"{execution_time:.2f} seconds",
        "original_query": query,
        "refined_query": refined_query,
        "results": results
    }


