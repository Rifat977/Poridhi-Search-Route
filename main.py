import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel


# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Step 1: Create and Save Embeddings ----------
def product_embedding(df):
    # Create embeddings from concatenated product attributes
    df['embedding'] = df.apply(lambda row: model.encode(
        f"{row['title']} {row['brand']} {row['description']} {row['final_price']} {row['availability']} {row['categories']} {row['item_weight']} {row['rating']}"
    ), axis=1)

    # Convert embeddings to 2D array
    embeddings = np.vstack(df['embedding'].values)

    # Drop embedding column before saving the raw product CSV
    df.drop(columns=['embedding'], inplace=True)
    df.to_csv("amazon-products.csv", index=False)

    # Create and save FAISS index
    vector_store = faiss.IndexFlatL2(embeddings.shape[1])
    vector_store.add(embeddings)
    faiss.write_index(vector_store, "amazon-products-embeddings.faiss")

    return embeddings

# ---------- Step 2: Search Similar Products ----------
def search_similar_products(query, top_k=5):
    # Encode the query
    query_embedding = model.encode([query])
    
    # Search the FAISS vector store
    distances, indices = vector_store.search(query_embedding, top_k)
    
    # Fetch product details by row indices
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = distances[0]
    return results

# ---------- Main Flow ----------

# Load product data (previously saved)
df = pd.read_csv("amazon-products.csv")

# Generate embeddings and FAISS index
# product_embedding(df)

# Load FAISS index
vector_store = faiss.read_index("amazon-products-embeddings.faiss")


app = FastAPI()

class ProductSearchQuery(BaseModel):
    query: str  


@app.get("/search/")
async def product_search(query: str):
    results = search_similar_products(query)

    # Replace NaN values with None (JSON serializable)
    results = results.replace({np.nan: None})

    # Convert to dict and cast similarity_score to float
    results_dict = results.to_dict(orient="records")
    for r in results_dict:
        if "similarity_score" in r and isinstance(r["similarity_score"], (np.float32, np.float64)):
            r["similarity_score"] = float(r["similarity_score"])

    return {"results": results_dict}
    



