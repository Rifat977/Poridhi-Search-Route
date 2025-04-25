import faiss, re, time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import google.generativeai as genai

from qdrant_client import QdrantClient
from qdrant_client.http import models


model = SentenceTransformer('all-MiniLM-L6-v2')

api_key = "AIzaSyBnyzxafKHtyuI98DEWg7xnO_h3qtJR2Nc"
genai.configure(api_key=api_key)
llm_model = genai.GenerativeModel("gemini-2.0-flash")

client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "amazon_products"

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=384,             
        distance=models.Distance.COSINE,
    ),

)

app = FastAPI()

df = pd.read_csv("data/final_data.csv")


# ---------- Step 1: Create and Save Embeddings ----------
# def product_embedding(df):
#     # Create embeddings from concatenated product attributes
#     df['embedding'] = df.apply(lambda row: model.encode(
#         # title,brand,description,final_price,availability,categories,item_weight,rating,model_number,department,length,width,height,price_bucket,primary_category
#         f"""
#             {row['title']} by {row['brand']}. 
#             Category: {row['primary_category']}. 
#             Description: {row['description']}. 
#             Price: ${row['final_price']} ({row['price_bucket']} price range). 
#             Customer Rating: {row['rating']} stars. 
#             Weight: {round(row['item_weight'], 2)}grams. 
#             Dimensions: {round(row['length'], 2)} cm x {round(row['width'], 2)} cm x {round(row['height'], 2)} cm. 
#             Department: {row['department']}. 
#         """

#     ), axis=1)

#     # Convert embeddings to 2D array
#     embeddings = np.vstack(df['embedding'].values)

#     # Drop embedding column before saving the raw product CSV
#     df.drop(columns=['embedding'], inplace=True)
#     df.to_csv("data/final_data_embedding.csv", index=False)
#     # Create and save FAISS index
#     vector_store = faiss.IndexFlatL2(embeddings.shape[1])
#     vector_store.add(embeddings)
#     faiss.write_index(vector_store, "data/amazon-products-embeddings.faiss")

#     return embeddings

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
        )
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
# def search_similar_products(query, top_k=5):
#     # Encode the query
#     query_embedding = model.encode([query])
    
#     # Search the FAISS vector store
#     distances, indices = vector_store.search(query_embedding, top_k)
    
#     # Fetch product details by row indices
#     results = df.iloc[indices[0]].copy()
#     results['similarity_score'] = distances[0]
#     return results

def search_similar_products(query: str, top_k: int = 5):
    # 1) Embed the incoming query
    q_vec = model.encode([query])[0]

    # 2) Ask Qdrant for the top_k most similar vectors
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vec,
        limit=top_k,
        with_payload=True,
        with_vectors=False,    # we don’t need the raw vectors back
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

# Load FAISS index
# vector_store = faiss.read_index("data/amazon-products-embeddings.faiss")


@app.get("/search/")
async def product_search(query: str):
    start_time = time.time()

    intent = extracts_intent_gemini(query)

    refined_query = intent.strip().strip('\'"')         
    refined_query = refined_query.replace("\\", "")    
    refined_query = re.sub(r'[^\w\s\-\.,%]', '', refined_query) 

    # results = search_similar_products(refined_query)

    # results = results.replace({np.nan: None})

    # results_dict = results.to_dict(orient="records")
    # for r in results_dict:
    #     if "similarity_score" in r and isinstance(r["similarity_score"], (np.float32, np.float64)):
    #         r["similarity_score"] = float(r["similarity_score"])

    # end_time = time.time()
    # execution_time = end_time - start_time
    

    # return {
    #     "execution_time": f"{execution_time:.2f} seconds", 
    #     "original_query": query,
    #     "refined_query": refined_query,
    #     "results": results_dict
    # }

    results_df = search_similar_products(refined_query, top_k=5)
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


