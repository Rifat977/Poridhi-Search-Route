from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import redis
import requests
import json

app = FastAPI()

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

class ProductSearchQuery(BaseModel):
    query: str

def search_in_vector_db(query: str, top_k: int = 10):
    pass

def get_query_embedding(query: str):
    return [0.1, 0.2, 0.3]  

def get_product_metadata(product_ids):
    response = requests.post("http://django-app/products/metadata", json={"ids": product_ids})
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error fetching product metadata")
    return response.json()

def get_from_cache(query: str):
    cached_result = redis_client.get(query)
    if cached_result:
        return json.loads(cached_result)
    return None

def set_to_cache(query: str, result: dict):
    redis_client.set(query, json.dumps(result), ex=3600)

@app.get("/search/")
async def product_search(query: str):
    cached_result = get_from_cache(query)
    if cached_result:
        return {"source": "cache", "products": cached_result}

    # vector_results = search_in_vector_db(query.query, query.top_k)

    # product_ids = [result['id'] for result in vector_results]
    # product_metadata = get_product_metadata(product_ids)

    search_result = query
    # for result in vector_results:
    #    metadata = next(item for item in product_metadata if item['id'] == result['id'])
    #   search_result.append({
    #       "id": result['id'],
    #       "name": metadata['name'],
    #       "description": metadata['description'],
    #       "price": metadata['price'],
    #       "image_url": metadata['image_url'],
    #       "score": result['score']  
#   })

    # set_to_cache(query.query, search_result)

    return {"source": "vector_db", "products": search_result}
